"""
In this version, we refine our protocol CNCP based on the previous version (cncp_v1).

To evaluate CNCP, we need to implement and simulate three type of nodes: (1) source nodes, (2) intermediate nodes, 
and (3) destination nodes.

For the source node, we use the DistPacketGenerator class to implement its functionality. It mainly 
includes: (1) sending packets in an allocated rate; (2) deciding whether can send a packet currently, and determining
the packets of which file should be sent.

For the intermediate node, we use the Router class to implement its functionality. It mainly includes three components: 
(1) link queues (implemented in Port class), (2) user queues, (3) transmitting the packets from a user queue to a link queue
in an allocated rate (implemented in Allocator class).

For the destination node, we use the PacketSink class to implement its functionality. it is responsible to recording 
some statistic information from incoming packets, and identifying the successful decoding of different files.

Another two important class is the RateControlMonitor class and the ResetMonitor class. RateControlMonitor aims to update the
rate following the congestion control algorithm. ResetMonitor is responsible to monitoring the state of a flow (arrive/leave), 
then resetting the bandwidth allocated to each active flow.
"""

import simpy
from typing_extensions import Tuple

from ns.packet.packet import Packet
from dataclasses import dataclass
from collections.abc import Callable
from collections import defaultdict as dd
import copy
from ns.port.wire import Wire
import numpy as np
from collections import defaultdict

MAX = 100  # indicating the user_queue size for each user at a router
buffer_size = 100  # indicating the link_queue size for each port at a router
# two parameters for rate update
DELTA = 0.5
GAMMA = 0.1


def const_size():
    """
    return the packet size, assume the encoded packets have the same size
    """
    return 1000.0  # in B


def cal_flows_for_each_port(routing_info, nports):
    """
    calculate the number of flows that use a port
    """
    port_to_flow = dict()
    for port_id in range(nports):
        num_flow = 0
        for _, ports in routing_info.items():
            if port_id in ports:
                num_flow += 1
        port_to_flow[port_id] = num_flow
    return port_to_flow


def user_queue_length_to_q(user_queue_length):
    """get the node price according to the state of user queue"""
    return (MAX - user_queue_length) / MAX


def link_queue_length_to_p(link_queue_length):
    """get the link price according to the state of link queue"""
    return link_queue_length / buffer_size


class Packet_U(Packet):
    def __init__(self,
                 time,
                 size,
                 packet_id,
                 realtime=0,
                 src="source",
                 dst="destination",
                 flow_id=0,
                 file_id=0,
                 payload=None):
        super().__init__(time, size, packet_id, realtime, src, dst, flow_id, payload)

        self.file_id = file_id  # indicate which file is transmitted currently


class Port:
    """
    Models an output port on a switch with a given rate and buffer size (in the number of packets), using 
    the simple tail-drop mechanism to drop packets.

    Parameters
    ----------
    env: simpy.Environment
        the simulation environment.
    port_id: int
        the id of this port in the switch.
    rate: float
        the bit rate of the port (0 for unlimited).
    buffer_size: int
        the link queue size for this port
    debug: bool
        If True, prints more verbose debug information.
    """
    def __init__(self,
                 env,
                 port_id: int,
                 rate: float,
                 buffer_size,
                 debug: bool = False) -> None:
        self.env = env
        self.port_id = port_id
        self.rate = rate
        self.buffer_size = buffer_size
        self.debug = debug

        self.out = None
        # the link queue for an output port
        self.queue = simpy.Store(self.env, capacity=buffer_size)

        self.action = env.process(self.run())

    def run(self):
        while True:
            packet = yield self.queue.get()
            yield self.env.timeout(8.0 * const_size() / self.rate)
            self.out.put(packet)

    def put(self, packet):
        if len(self.queue.items) >= self.buffer_size:
            return
        self.queue.put(packet)


class Allocator:
    """
    Schedules packets from a user queue to a link queue using the allocated rate.

    Parameters
    ----------
    env: simpy.Environment
        the simulation environment.
    router: Router objective
        the router objective that the allocator belongs to.
    flow: int
        the flow id of the corresponding user queue from which the allocator get the packets.
    port: Port objective
        the port which the allocator send packets to.
    debug: bool
        If True, prints more verbose debug information.
    """
    def __init__(self,
                 env,
                 router,
                 flow,
                 init_f,
                 max_f,
                 port,
                 debug: bool = False) -> None:
        self.env = env
        self.router = router
        self.flow = flow
        self.port = port
        self.debug = debug

        self.can_update = False  # add

        # transfer the rate in the unit of "us/ms sending a packet"
        self.init_f = 8.0 * const_size() / init_f  # add
        self.f = 8.0 * const_size() / init_f
        self.max_f = 8.0 * const_size() / max_f

        self.action = env.process(self.run())

    def reset_f(self):
        self.f = self.init_f

    def run(self):
        while True:
            # adopt the LIFO manner in the user queue
            if len(self.router.user_queues[self.flow].items) > 0:
                packet = self.router.user_queues[self.flow].items.pop()
                self.port.put(packet)
            yield self.env.timeout(self.f)

            # packet = yield self.router.user_queues[self.flow].get()
            # yield self.env.timeout(self.f)
            # self.port.put(packet)


class Router:
    """
    Models an intermediate node (router/switch) that includes the user queues, link queues, and allocators.

    Parameters
    ----------
    env: simpy.Environment
        the simulation environment.
    debug: bool
        If True, prints more verbose debug information.
    routing_info: dict
        key: the flow id that use this router, value: the port_ids that the flow uses.
    port_rate: float
        the bit rate of all ports belonging to this router.
    nports: int
        the number of ports in this router.
    """
    def __init__(self,
                 env,
                 nports: int,
                 port_rate: float,
                 buffer_size: int,
                 routing_info: dict,
                 id: int,
                 ends=None,
                 debug: bool = False) -> None:
        self.env = env
        self.debug = debug
        self.id = id

        # an example of routing_info: {0: [1, 2], ...}.
        # it means that the packets of flow with flow_id 0 will use port 1 and 2 in this router
        self.routing_info = routing_info
        self.port_rate = port_rate
        self.nports = nports

        # three main components in a router
        self.ports = []
        self.user_queues = dict()
        # an allocator corresponds to a "line" from the user queue to a link queue
        self.allocators = []

        # initialize the ports
        for port in range(nports):
            self.ports.append(
                Port(env,
                     rate=port_rate,
                     port_id=port,
                     buffer_size=buffer_size,  # in packets
                     debug=debug))

        if ends:
            self.ends = ends
        else:
            self.ends = defaultdict(dict)

        self.init_settings()

    def init_settings(self):
        """
        initialize the components belonging to a router, including the user_queues and
        allocators. Note that to initialize the user_queues which are related
        with the flows, we should first define the routing information (routing_info).
        """
        port_to_flow = cal_flows_for_each_port(self.routing_info, self.nports)

        for flow in self.routing_info.keys():
            self.user_queues[flow] = simpy.Store(self.env, capacity=MAX)

            # for each port that is included by a flow
            for port in self.routing_info[flow]:
                # the bandwidth for each port is equally divided for each flow
                f = self.port_rate / port_to_flow[port]
                f_max = self.port_rate
                self.allocators.append(Allocator(env=self.env,
                                                 router=self,
                                                 flow=flow,
                                                 init_f=f,
                                                 max_f=f_max,
                                                 port=self.ports[port],
                                                 debug=self.debug
                                                 ))

    def put(self, packet):
        flow_id = packet.flow_id
        if flow_id in self.ends:
            self.ends[flow_id].put(packet)
        else:
            if len(self.user_queues[flow_id].items) >= MAX:
                return
            self.user_queues[flow_id].put(packet)


@dataclass
class Flow:
    """ A dataclass for keeping track of all the properties of a network flow. """
    fid: int  # flow id
    src: str  # source element
    dst: str  # destination element
    size: int = None  # flow size in bytes
    start_time: float = None
    finish_time: float = None
    arrival_dist: Callable = None  # packet arrival distribution
    size_dist: Callable = None  # packet size distribution
    pkt_gen: object = None   # the corresponding source node of the flow
    pkt_sink: object = None  # the corresponding destination node of the flow
    path: list = None
    experi_intermediate_nodes: list = None  # the intermediate nodes experienced by the flow

    def __repr__(self) -> str:
        return f"Flow {self.fid} on {self.path}"


class Flow_U(Flow):
    """
    Associate the file information for the flow.

    Parameters
    ----------
    num_files: int
        the total number of files that need to be transmitted.
    file_done: list
        indicate whether is completed transmission for each file.
    done_times: list
        record the finish time for each file.
    actual_file_arrival_time: list
        record the actual arrival time for each file.
    """
    def __init__(self,
                 fid,
                 src,
                 dst,
                 size,   # the size for each file
                 delay_interval):    # record the intervals for consecutive files
        super().__init__(fid,src,dst,size)

        # record all information of multiple files
        # the number of files
        self.num_files = len(delay_interval) + 1
        self.file_done = [False for _ in range(self.num_files)]
        # the completion time of each file
        self.done_times = [None for _ in range(self.num_files)]

        self.actual_file_arrival_time = [0]
        tmp_tot = 0
        for i in range(len(delay_interval)):
            tmp_tot += delay_interval[i]
            self.actual_file_arrival_time.append(tmp_tot)


class DistPacketGenerator:
    """
    Generates packets for different files in a FIFO manner and following the constraints of arrival time of files.

    Parameters
    ----------
    env: simpy.Environment
        the simulation environment.
    flow: Flow_U objective
        all files for a source-destination pair have the same flow id.
    arrival_dist:
        control the sending rate from the source node.
    size_dist:
        packet size.
    packets_sent: int
        record the number of packets that have been sent.
    debug: bool
        If True, prints more verbose debug information.
    """
    def __init__(self,
                 env,
                 flow,
                 id: int,
                 arrival_dist,
                 size_dist,   # the packet size
                 debug=False):
        self.env = env
        self.flow = flow
        self.id = id
        self.arrival_dist = arrival_dist
        self.size_dist = size_dist
        self.debug = debug

        self.packets_sent = 0
        self.out = None

        self.action = env.process(self.run())

    def run(self):
        while True:
            # wait for next transmission
            yield self.env.timeout(self.arrival_dist())
            can_send = self.check_available()
            if can_send:
                self.packets_sent += 1
                file_id = self.determine_file_to_send()

                packet = Packet_U(self.env.now,
                                  self.size_dist(),
                                  self.packets_sent,
                                  flow_id=self.flow.fid,
                                  file_id=file_id)

                if self.debug:
                    print(
                        "Sent packet (packet_id:{:d}, file_id:{:d}, flow {:d}) at time {:.4f}.".format(
                            packet.packet_id, packet.file_id, packet.flow_id,
                            self.env.now))

                self.out.put(packet)

    def check_available(self):
        """determine whether can send a new packet currently (no matter which file)"""
        now = self.env.now

        cur_order = None  # file id that should be transmitted
        if all(self.flow.file_done):
            return False

        if not any(self.flow.file_done):  # no file has completed transmission
            cur_order = 0
        else:
            for i in range(len(self.flow.file_done) - 1):
                if self.flow.file_done[i] == True and self.flow.file_done[i + 1] == False:
                    cur_order = i + 1

        init_t = self.flow.actual_file_arrival_time[cur_order]
        cur_done = self.flow.file_done[cur_order]
        if cur_order == 0:
            past_done = True
        else:
            past_done = self.flow.file_done[cur_order - 1]

        if ((now >= init_t) and (past_done == True) and (cur_done == False)) or \
                ((now <= init_t) and (past_done == False) and (cur_done == False)) or \
                ((now >= init_t) and (past_done == False) and (cur_done == False)):
            return True
        else:
            return False

    def determine_file_to_send(self):
        """determine the packets of which file can be sent currently.

        Returns
        -------
        cur_order: the file id of which the packets can be sent
        """
        cur_order = None
        if not any(self.flow.file_done):
            cur_order = 0
        else:
            for i in range(len(self.flow.file_done) - 1):
                if self.flow.file_done[i] == True and self.flow.file_done[i + 1] == False:
                    cur_order = i + 1
        return cur_order


class PacketSink:
    """
    A PacketSink is designed to record arrival times and the rate achieved from the incoming packets, and identify the 
    successful decoding of current file.

    Parameters
    ----------
    env: simpy.Environment
        the simulation environment.
    flow: Flow_U objective
        all files for a source-destination pair have the same flow id.
    inst_rate: float
        the achieved instantaneous rate.
    rates: list
        record all inst_rate.
    waits: dictionary
        record the waiting times experienced by the packets
    packet_times: dictionary
        record the sending time for the packet that arrives the destination successfully.
    packet_sizes: dictionary
        record the total size of packets for each file
    record_arrival_pkt_num: dictionary
        record the total number of arrival packets of different files.
    debug: bool
        If True, prints more verbose debug information.
    """
    def __init__(self,
                 env,
                 id,
                 flow,
                 debug: bool = False):
        self.env = env
        self.id = id
        self.flow = flow

        self.inst_rate = 0
        self.rates = list()
        self.waits = dd(list)
        self.packet_times = dd(list)
        self.packet_sizes = dd(list)

        self.record_arrival_pkt_num = dict()

        self.debug = debug

        for i in range(self.flow.num_files):
            self.record_arrival_pkt_num[i] = 0

    def put(self, packet):
        """On receiving a coded packet."""
        now = self.env.now
        # identify which file the packet belongs to
        rec_index = packet.file_id

        self.record_arrival_pkt_num[packet.file_id] += 1

        self.waits[rec_index].append(self.env.now - packet.time)
        self.packet_sizes[rec_index].append(packet.size)
        self.packet_times[rec_index].append(packet.time)

        if self.debug:
            print("At time {:.2f}, packet (packet_id:{:d}, file_id:{:d}, flow {:d}) arrived.".format(
                now, packet.packet_id, packet.file_id, packet.flow_id,))

            if len(self.packet_sizes[rec_index]) >= 10:
                bytes_received = sum(self.packet_sizes[rec_index][-9:])
                time_elapsed = self.env.now - (
                        self.packet_times[rec_index][-10] +
                        self.waits[rec_index][-10])
                if self.debug:
                    print(
                        "Average throughput (last 10 packets): {:.2f} bytes/second."
                        .format(float(bytes_received) / time_elapsed))
                self.inst_rate = float(bytes_received) / time_elapsed
                self.rates.append([self.env.now, copy.deepcopy(self.inst_rate) / 1000 * 8])

        # if the number of received coded packets of current file is larger than the original file size,
        # it implies that current file can be decoded correctly.
        if self.record_arrival_pkt_num[rec_index] >= self.flow.size:
            if not self.flow.file_done[rec_index]:
                self.flow.file_done[rec_index] = True
                self.flow.done_times[rec_index] = self.env.now
                if self.debug:
                    print("=====file {:d} is successfully decoded at sink {:d} at time {:.4f}=====".format(
                        rec_index, self.id, self.env.now))


class RateControlMonitor:
    """
    In our CNCP_v2 implementation, we simulate the feedback between two adjacent nodes by 
    the RateControlMonitor to decrease the complexity. The RateControlMonitor will be called 
    each time the rate can be updated, this is controlled by the property "self.dist".
    Note that in the rate calculation, the unit of rate is in the form of "packet/ms", i.e., how 
    many packets can be sent in 1 ms(us).

    Parameters
    ----------
    env: simpy.Environment
        the simulation environment.
    root_router: Router objective
        the upstream node (router/switch)
    target_routers: list of Router objectives
        all downstream nodes (router/switch) of root_router
    """
    def __init__(self,
                 env,
                 root_router,
                 id: int,
                 dist,
                 target_routers: list = None):
        self.env = env
        self.root_router = root_router
        self.id = id
        if target_routers:
            self.target_routers = target_routers
        else:
            self.target_routers = []

        # Currently, the time interval of two consecutive rate update should equal to
        # the one-hop RTT between the two corresponding routers
        self.dist = dist

        # self.data = dict()
        # for allocator in self.root_router.allocators:
        #     port_id = allocator.port.port_id
        #     flow_id = allocator.flow
        #     self.data[f"{root_router.id}-{port_id}-{flow_id}"] = []

        self.action = env.process(self.run())

    def run(self):
        while True:
            yield self.env.timeout(self.dist())
            if not self.target_routers:
                # the target_routers is empty
                continue
            for target_router in self.target_routers:
                # the next-hop is the destination
                if isinstance(target_router, PacketSink):
                    root_allocators = self.root_router.allocators
                    for allocator in root_allocators:
                        if allocator.can_update:
                            dest_rate = target_router.inst_rate / 1000  # in the unit of "KB (packet)/ms"
                            if dest_rate == 0:
                                dest_rate = 0.125 / 1  # packets / ms

                            check_port = allocator.port.port_id
                            check_flow = allocator.flow

                            # pay attention!!!!!!
                            if self.root_router.ports[check_port].out.out.id != target_router.id:
                                continue

                            root_user_queue_length = user_queue_length_to_q(
                                len(self.root_router.user_queues[check_flow].items))
                            root_port_queue_length = link_queue_length_to_p(
                                len(self.root_router.ports[check_port].queue.items))

                            if allocator.f == 0:
                                allocator.f = 10
                            # the unit of allocator.f is "ms/packet"
                            cur_f = 1 / allocator.f  # packets / ms

                            new_f = cur_f + DELTA * (1 / dest_rate) + GAMMA * (
                                    - root_user_queue_length - root_port_queue_length)

                            # the maximal value of updated rate is the port rate
                            if new_f > 1 / allocator.max_f:
                                new_f = 1 / allocator.max_f

                            if new_f <= 0:
                                new_f = 0.125
                            allocator.f = copy.deepcopy(1 / new_f)
                            # self.data[f"{self.root_router.id}-{check_port}-{check_flow}"].append([self.env.now, new_f])
                        # else:
                            # self.data[f"{self.root_router.id}-{allocator.port.port_id}-{allocator.flow}"].append(
                            #     [self.env.now, 0])
                else:
                    root_allocators = self.root_router.allocators
                    for allocator in root_allocators:
                        if allocator.can_update:  # add
                            check_port = allocator.port.port_id
                            check_flow = allocator.flow

                            # similar to the above case
                            if self.root_router.ports[check_port].out.out.id != target_router.id:
                                continue

                            root_user_queue_length = user_queue_length_to_q(
                                len(self.root_router.user_queues[check_flow].items))
                            # In this place, we also simplify the operation for fattree_cncp_v2,
                            # when the next_hop has no user queue for the corresponding flow, it implies that this is the last hop,
                            # and we assume the user queue is fully empty (target_user_queue_length = 1)
                            if check_flow in target_router.user_queues:
                                target_user_queue_length = user_queue_length_to_q(
                                    len(target_router.user_queues[check_flow].items))
                            else:
                                target_user_queue_length = 1
                            root_port_queue_length = link_queue_length_to_p(
                                len(self.root_router.ports[check_port].queue.items))

                            if allocator.f == 0:
                                allocator.f = 10
                            cur_f = 1 / allocator.f

                            new_f = cur_f + 0.1 * ((1.1 * -root_port_queue_length) - root_user_queue_length +
                                                   target_user_queue_length) + 0.05 * (1 / cur_f)
                            
                            # the maximal value of updated rate is the port rate
                            if new_f > 1 / allocator.max_f:
                                new_f = 1 / allocator.max_f

                            if new_f <= 0:
                                new_f = 0.125
                            allocator.f = copy.deepcopy(1 / new_f)
                            # self.data[f"{self.root_router.id}-{check_port}-{check_flow}"].append([self.env.now, new_f])
                        # else:
                            # self.data[f"{self.root_router.id}-{allocator.port.port_id}-{allocator.flow}"].append(
                            #     [self.env.now, 0])


class ResetMonitor:
    """
    In our CNCP_v2 implementation, we use ResetMonitor to monitor the state of a flow (arrive/leave), then reset
    the bandwidth allocated to each active flow. In reality, the user queue for a flow will disappear if the flow 
    leaves, and the bandwidth will be recycled.

    Parameters
    ----------
    env: simpy.Environment
        the simulation environment.
    source: the DistPacketGenerator objective
        the source node of a flow. Each flow need such a ResetMonitor.
    routers: list of Router objectives
        the intermediate nodes experienced by a flow. 
    """
    def __init__(self,
                 env,
                 source,
                 routers,
                 id: int,
                 dist):

        self.source = source
        self.routers = routers
        self.env = env
        self.id = id
        self.dist = dist

        self.action = env.process(self.run())

    def run(self):
        while True:
            yield self.env.timeout(self.dist())
            can_send = self.source.check_available()

            # delete the packets of the previous file
            # when file size is large, this part can be excluded
            # file_id_to_send = self.source.determine_file_to_send()
            # if file_id_to_send > 0:
            #     file_id_to_del = file_id_to_send - 1

            # for each router used by the source (flow)
            for router in self.routers:
                # for each port of the router
                for port_id in range(len(router.ports)):
                    port_obj = router.ports[port_id]
                    port_total_rate = port_obj.rate
                    # get number of active flows
                    self_active = 0
                    for allocator in router.allocators:
                        if allocator.port.port_id == port_id:
                            if allocator.flow == self.source.flow.fid:
                                if can_send:
                                    self_active = 1

                    if self_active != 0:
                        # check whether to update
                        should_update = False
                        for allocator in router.allocators:
                            if allocator.port.port_id == port_id:
                                if allocator.flow == self.source.flow.fid:
                                    if allocator.can_update == False and can_send == True:
                                        # new flow incoming
                                        should_update = True
                                        break

                        # update state
                        if should_update:
                            for allocator in router.allocators:
                                if allocator.port.port_id == port_id:
                                    if allocator.flow == self.source.flow.fid:
                                        allocator.can_update = True

                        # reset f
                        if should_update:
                            num_active_flows = 0
                            for allocator in router.allocators:
                                if allocator.port.port_id == port_id:
                                    if allocator.can_update:
                                        num_active_flows += 1

                            reset_rate = port_total_rate / num_active_flows
                            for allocator in router.allocators:
                                if allocator.port.port_id == port_id:
                                    allocator.f = 8.0 * const_size() / reset_rate

                    # check if flow leaves
                    for allocator in router.allocators:
                        if allocator.port == port_obj:
                            if allocator.flow == self.source.flow.fid:
                                if allocator.can_update == True and can_send == False:
                                    allocator.can_update = False


#######################################################################################
# construct a toy-example to test cncp. The topology is:
# s_1                                 d_1
#     --                          --
#        --                   --
#            r_1 -------- r_2
#        --                   --
#     --                          -- 
# s_2                                 d_2


def source_arrival_100():
    return 0.08

def delay_dist_monitor():
    return 0.2

def delay_dist():
    """ Network wires experience a constant propagation delay of 0.1 microseconds. """
    return 0.1

if __name__ == "__main__":
    # the time unit of this environment is "1ms"
    env = simpy.Environment()
    rate_100 = 8.0 * const_size() / 0.08  # 100Mbps

    file_num = 20
    interval = 20
    # files can arrive at the source following a Poisson distribution
    # for testing, we assume the arrival of files is constant
    src_interval_delays = [interval for _ in range(file_num)]

    flow_one = Flow_U(fid=0,
                  src='flow 1',
                  dst='flow 1',
                  size=100,
                  delay_interval=src_interval_delays)

    flow_two = Flow_U(fid=1,
                      src='flow 2',
                      dst='flow 2',
                      size=100,
                      delay_interval=src_interval_delays)

    src_one = DistPacketGenerator(env, flow=flow_one, id=0, arrival_dist=source_arrival_100,
                                  size_dist=const_size, debug=True)
    src_two = DistPacketGenerator(env, flow=flow_two, id=1, arrival_dist=source_arrival_100,
                                  size_dist=const_size, debug=True)

    dest_one = PacketSink(env, flow=flow_one, id=0, debug=True)
    dest_two = PacketSink(env, flow=flow_two, id=1, debug=True)

    wire1 = Wire(env, delay_dist)
    wire2 = Wire(env, delay_dist)
    wire3 = Wire(env, delay_dist)
    wire4 = Wire(env, delay_dist)
    wire5 = Wire(env, delay_dist)

    # routing_info={0:[0], 1:[0]} implies that both flow 0 and 1 will use port 0 of router_one
    router_one = Router(env, nports=1, port_rate=rate_100, buffer_size=100,
                        routing_info={0:[0], 1:[0]}, id=0, debug=True)

    router_two = Router(env, nports=2, port_rate=rate_100, buffer_size=100,
                        routing_info={0: [0], 1: [1]}, id=1, debug=True)

    src_one.out = wire1
    wire1.out = router_one
    src_two.out = wire2
    wire2.out = router_one
    router_one.ports[0].out = wire3

    wire3.out = router_two
    router_two.ports[0].out = wire4
    router_two.ports[1].out = wire5
    wire4.out = dest_one
    wire5.out = dest_two

    rate_monitor_one = RateControlMonitor(env, root_router=router_one,
                                          target_routers=[router_two], id = 0, dist=delay_dist_monitor)

    rate_monitor_two = RateControlMonitor(env, root_router=router_two,
                                          target_routers=[dest_one, dest_two], id=1, dist=delay_dist_monitor)

    reset_monitor_one = ResetMonitor(env, source=src_one, routers=[router_one, router_two],
                                     id=0, dist=delay_dist_monitor)

    reset_monitor_two = ResetMonitor(env, source=src_two, routers=[router_one, router_two],
                                     id=1, dist=delay_dist_monitor)

    env.run(until=400)

    dest1x = []
    dest1y = []
    for i in range(len(dest_one.rates)):
        if i % 5 == 0:
            dest1x.append(dest_one.rates[i][0])
            dest1y.append(dest_one.rates[i][1])

    dest2x = []
    dest2y = []
    for i in range(len(dest_two.rates)):
        if i % 5 == 0:
            dest2x.append(dest_two.rates[i][0])
            dest2y.append(dest_two.rates[i][1])

    print("=====source one=====")
    print(flow_one.file_done)
    print(flow_one.actual_file_arrival_time)
    print(flow_one.done_times)

    fct_one = list()
    for i in range(len(flow_one.done_times)):
        x = flow_one.actual_file_arrival_time[i]
        y = flow_one.done_times[i]
        if y is not None:
            fct_one.append(y - x)
    avg_fct_one = np.average(fct_one)
    print(fct_one)
    print(avg_fct_one)

    print("=====source two=====")
    print(flow_two.file_done)
    print(flow_two.actual_file_arrival_time)
    print(flow_two.done_times)

    fct_two = list()
    for i in range(len(flow_two.done_times)):
        x = flow_two.actual_file_arrival_time[i]
        y = flow_two.done_times[i]
        if y is not None:
            fct_two.append(y - x)
    avg_fct_two = np.average(fct_two)
    print(fct_two)
    print(avg_fct_two)

    # print(dest_one.record_arrival_pkt_num)
    # print(dest_two.record_arrival_pkt_num)

    # f1x = []
    # f1y = []
    # for i in range(len(rate_monitor_one.data['0-0-0'])):
    #     f1x.append(rate_monitor_one.data['0-0-0'][i][0])
    #     f1y.append(rate_monitor_one.data['0-0-0'][i][1] * 8)
    #
    # f2x = []
    # f2y = []
    # for i in range(len(rate_monitor_one.data['0-0-1'])):
    #     f2x.append(rate_monitor_one.data['0-0-1'][i][0])
    #     f2y.append(rate_monitor_one.data['0-0-1'][i][1] * 8)

    # import matplotlib.pyplot as plt

    # plt.plot(f1x, f1y, label="f1")
    # plt.plot(f2x, f2y, label='f2')

    # plt.plot(dest1x, dest1y)
    # plt.plot(dest2x, dest2y)
    # plt.show()