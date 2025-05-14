"""
In this script, we use the components defined in cncp_v2 to construct the scenarios
where flows are transmitted on the fat_tree topology. 
In this scenario, each flow uses a single path.  
"""
import copy

import networkx as nx
import simpy
from random import sample
import random
from collections import defaultdict
import numpy as np
from scipy.stats import truncnorm
from cncp_v2 import (DistPacketGenerator, PacketSink, Router,
                     ResetMonitor, RateControlMonitor, Flow_U)
from ns.port.wire import Wire
import csv
import time

def generate_truncated_normal(mean, variance, lower, upper, size, random_seed):
    values = []
    std_dev = np.sqrt(variance)
    np.random.seed(random_seed)
    while len(values) < size:
        value = np.random.normal(mean, std_dev)
        if lower <= value <= upper:
            values.append(value)
    rounded_numbers = [round(num) for num in values]
    return rounded_numbers


def truncated_gaussian(mean, variance, lower, upper, size, random_seed):
    stddev = np.sqrt(variance)
    a, b = (lower - mean) / stddev, (upper - mean) / stddev
    np.random.seed(random_seed)
    sample = truncnorm.rvs(a, b, loc=mean, scale=stddev, size=int(size))
    rounded_numbers = [round(num) for num in sample]
    return rounded_numbers


def truncated_exponential(scale, lower, upper, random_seed, size=None):
    samples = []
    np.random.seed(random_seed)
    while len(samples) < (size if size else 1):
        sample = np.random.exponential(scale=scale)
        if lower <= sample <= upper:
            samples.append(sample)
    return samples if size else samples[0]


def const_size():
    return 1000.0  # in B

def propagation_delay_dist():
    """ Network wires experience a constant propagation delay of 1 microsecond. """
    return 1

def source_arrival_100():
    return 0.08

def delay_dist_monitor():
    return 2  # it should equal to the one-hop RTT

def delay_dist_monitor_reset():
    return 0.1

def delay_dist_monitor_dynamics():
    return 0.1

def delay_dist_output_monitor():
    return 10

def build(k):
    # validate input arguments
    if not isinstance(k, int):
        raise TypeError('k argument must be of int type')
    if k < 1 or k % 2 == 1:
        raise ValueError('k must be a positive even integer')

    topo = nx.Graph()
    topo.name = "fat_tree_topology(%d)" % (k)

    # Create core nodes
    n_core = (k // 2)**2
    topo.add_nodes_from([v for v in range(int(n_core))],
                        layer='core',
                        type='switch')

    # Create aggregation and edge nodes and connect them
    # for each pod
    # each pod has k/2 aggregation switches, and k/2 edge switches
    for pod in range(k):
        aggr_start_node = topo.number_of_nodes()  # return the number of nodes in the graph
        aggr_end_node = aggr_start_node + k // 2
        edge_start_node = aggr_end_node
        edge_end_node = edge_start_node + k // 2
        aggr_nodes = range(aggr_start_node, aggr_end_node)
        edge_nodes = range(edge_start_node, edge_end_node)
        topo.add_nodes_from(aggr_nodes,
                            layer='aggregation',
                            type='switch',
                            pod=pod)
        topo.add_nodes_from(edge_nodes, layer='edge', type='switch', pod=pod)
        topo.add_edges_from([(u, v) for u in aggr_nodes for v in edge_nodes],
                            type='aggregation_edge')

    # Connect core switches to aggregation switches
    for core_node in range(n_core):
        for pod in range(k):
            aggr_node = n_core + (core_node // (k // 2)) + (k * pod)
            topo.add_edge(core_node, aggr_node, type='core_aggregation')

    # Create hosts and connect them to edge switches
    for u in [v for v in topo.nodes() if topo.nodes[v]['layer'] == 'edge']:
        leaf_nodes = range(topo.number_of_nodes(),
                           topo.number_of_nodes() + k // 2)
        topo.add_nodes_from(leaf_nodes,
                            layer='leaf',
                            type='host',
                            pod=topo.nodes[u]['pod'])
        topo.add_edges_from([(u, v) for v in leaf_nodes], type='edge_leaf')

    return topo

def get_experi_intermediate_nodes(path):
    intermediate_nodes = []
    for subpath in path:
        subpath_ = subpath[1:-1]
        for hop in subpath_:
            if hop not in intermediate_nodes:
                intermediate_nodes.append(hop)
    return intermediate_nodes

def gen_src_dest_pair(hosts, nflows):
    """
    generate the src-dest pairs that satisfy our requirements, i.e., sampling src-dst pairs without replacement
    Parameters
    ----------
    hosts: the server set in the fat_tree network
    nflows: the number of src-dest pairs (flows) that should be sampled
    """
    if len(hosts) < 2 * nflows:
        raise ValueError("does not have enough hosts!")
    host_list = list(hosts)
    src_dest_pairs = []
    random.seed(0)
    sampled_elements = random.sample(host_list, 2 * nflows)
    for i in range(nflows):
        src = sampled_elements[2 * i]
        dest = sampled_elements[2 * i + 1]
        src_dest_pairs.append((src, dest))
    return src_dest_pairs

def read_src_dest_pairs(file_path):
    tuples_list = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip().rstrip(',')
            line = line.strip('()')
            if line:
                tuple_data = tuple(map(int, line.split(',')))
                tuples_list.append(tuple_data)
    return tuples_list

def read_flow_path(file_path):
    flow_path = dict()
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            index, list_str = line.split(',', 1)
            ind = eval(index)
            number_list = eval(list_str)
            flow_path[ind] = number_list
    return flow_path

def generate_flows(
    G,
    hosts,
    nflows,
    delay_intervals,
    size=None
    ):
    all_flows = dict()
    # read the src_dst pairs
    src_dest_pairs = read_src_dest_pairs('src_dest_pair.txt')
    flow_path = read_flow_path("flow_path.txt")
    for flow_id in range(nflows):
        src, dest = src_dest_pairs[flow_id][0], src_dest_pairs[flow_id][1]
        all_flows[flow_id] = Flow_U(
            fid=flow_id,
            src=src,
            dst=dest,
            size=size,
            delay_interval=delay_intervals[flow_id]
        )
        all_flows[flow_id].path = [flow_path[flow_id]]
        all_flows[flow_id].experi_intermediate_nodes = get_experi_intermediate_nodes(all_flows[flow_id].path)
    return all_flows

def change_path_to_segment(path):
    segment = defaultdict(list)
    for subpath in path:
        for seg in subpath:
            a, z = seg
            if z not in segment[a]:
                segment[a].append(z)
    return segment

def generate_fib(G, all_flows):
    for n in G.nodes():
        node = G.nodes[n]

        node["port_to_nexthop"] = dict()
        node["nexthop_to_port"] = dict()

        # for a node, the number of ports equals to the number of adjacent nodes
        for port, nh in enumerate(nx.neighbors(G, n)):
            node["nexthop_to_port"][nh] = port
            node["port_to_nexthop"][port] = nh

        node["flow_to_port"] = defaultdict(list)
        node["flow_to_nexthop"] = defaultdict(list)
        node["nexthop"] = []

    for f in all_flows:
        flow = all_flows[f]
        path_hop = []
        for subpath in flow.path:
            path_hop.append(list(zip(subpath, subpath[1:])))
        segment = change_path_to_segment(path_hop)
        for seg in segment:
            next_hops = segment[seg]
            for one_candidate_hop in next_hops:
                G.nodes[seg]["flow_to_port"][flow.fid].append(G.nodes[seg]["nexthop_to_port"][one_candidate_hop])
                G.nodes[seg]["flow_to_nexthop"][flow.fid].append(one_candidate_hop)
                if one_candidate_hop not in G.nodes[seg]["nexthop"]:
                    G.nodes[seg]["nexthop"].append(one_candidate_hop)
    return G


class output_Monitor:
    def __init__(self,
                 env,
                 dist,
                 ):
        self.env = env
        self.dist = dist
        self.action = env.process(self.run())
    
    def run(self):
        while True:
            yield self.env.timeout(self.dist())
            print("current time:", self.env.now)


# each port of the router will independently follow the pattern for bandwidth modification
class modify_bandwidth_monitor:
    def __init__(self,
                 env,
                 router,
                 random_seed,
                 change_time,  # it indicates how many times it will sample from the exponential dist for each port.
                 variance,
                 id: int,
                 dist):

        self.router = router
        self.nports = router.nports
        self.random_seed = random_seed
        self.env = env
        self.id = id
        self.dist = dist

        # the scale is the \theta parameter in the paper
        self.time = truncated_exponential(scale=10, lower=1, upper=100, size=change_time * self.nports,
                                          random_seed=self.random_seed)
        # self.bandwidth_for_short = truncated_gaussian(mean=20, variance=25, lower=15, upper=25, size=change_time * self.nports,
        #                                               random_seed=self.random_seed)
        self.bandwidth_for_short = generate_truncated_normal(mean=20, variance=variance, lower=15, upper=25, size=change_time * self.nports,
                                                      random_seed=self.random_seed)
        self.actual_change_time = defaultdict(list)
        self.bandwidth_for_short_change = defaultdict(list)
        # we define the dynamics for each port
        for i in range(self.nports):
            tmp_time = 0
            for j in range(i*change_time, (i+1)*change_time):
                tmp_time += self.time[j]
                self.actual_change_time[i].append(tmp_time)
                self.bandwidth_for_short_change[i].append(self.bandwidth_for_short[j])

        self.update_time = {port_id: 0 for port_id in range(self.nports)}

        # print(self.bandwidth_for_short_change)

        self.action = env.process(self.run())

    def run(self):
        while True:
            yield self.env.timeout(self.dist())
            for port_id in range(self.nports):
                if self.env.now < self.actual_change_time[port_id][self.update_time[port_id]]:
                    continue
                else:
                    # print(self.router.id, self.bandwidth_for_short_change[port_id][self.update_time[port_id]])
                    time_each_packet = 8 / (100 - self.bandwidth_for_short_change[port_id][self.update_time[port_id]])
                    self.router.ports[port_id].rate = 8.0 * const_size() / time_each_packet
                    self.update_time[port_id] += 1

# To test the performance limit of all protocols, we construct a static network state
class static_construction_monitor:
    def __init__(self,
                 env,
                 router,
                 id: int,
                 bandwidth_for_short,
                 dist):
        self.router = router
        self.nports = router.nports
        self.env = env
        self.id = id
        self.dist = dist
        self.bandwidth_for_short = bandwidth_for_short

        self.action = env.process(self.run())

    def run(self):
        while True:
            yield self.env.timeout(self.dist())
            for port_id in range(self.nports):
                time_each_packet = 8 / (100 - self.bandwidth_for_short)
                self.router.ports[port_id].rate = 8.0 * const_size() / time_each_packet

# Note that the rate from the last switch to the destination is always the maximal rate
# since no other flows compete for the bandwidth (we sample the src-dest pair without replacement); Further, the maximal
# rate from the source is the same as the maximal rate to the destination (e.g., 100Gbps, 400Gbps). Hence, we can say
# that, the use queue at the last switch is always empty, and we do not construct the rateControlMonitor for rate
# update over the last_hop (from the last switch to the destination).
if __name__ == "__main__":
    start_time = time.time()
    test_id = input("test ID: ")
    print("initializing---")
    # the time unit of this environment is "1us"
    env = simpy.Environment()
    finish_time = 100000  # the total running time for the simulation, 100ms

    rate_100 = 8.0 * const_size() / 0.08  # 100Gbps

    n_flows = 32  # the total number of flows in the simulation
    k = 8   # the hyperparameter to build the fat_tree topology
    buffer_size = 100  # the buffer size for each port in the switches

    # indicate whether the arrival pattern of files for each flow follows poisson distribution
    # 0 means constant arrival, 1 means poisson arrival
    is_Poisson = 0

    theta = 10
    variance = 25

    file_size = 512000  # the file size for each file (in packets), in DCNs, 512MB
    file_num = 10   # the total number of files for each flow
    interval = 100000  # the interval is 100ms
    # files can arrive at the source following a Poisson distribution
    # for testing, we assume the arrival of files is constant
    interval_delays = [interval for _ in range(file_num)]
    total_interval_delays = [interval_delays for _ in range(n_flows)]

    # the change time that should be set, which is related with the running time of the whole system.
    # Usually, CHANGE_TIME * \theta (the first parameter of truncated_exponential function) should
    # larger than the running time of the whole system.
    CHANGE_TIME = int((finish_time / theta) + 10000)

    # record_rate = []

    # construct the fat_tree topology
    ft = build(k)

    hosts = set()
    for n in ft.nodes():
        if ft.nodes[n]["type"] == "host":
            hosts.add(n)

    all_flows = generate_flows(
        ft,
        hosts,
        n_flows,
        total_interval_delays,
        file_size
    )

    for fid in all_flows:
        cncp_src = DistPacketGenerator(env,
                                       flow=all_flows[fid],
                                       id=fid,
                                       arrival_dist=source_arrival_100,
                                       size_dist=const_size,
                                       debug=False)
        cncp_dst = PacketSink(env,
                              flow=all_flows[fid],
                              id=fid,
                              debug=False)
        all_flows[fid].pkt_gen = cncp_src
        all_flows[fid].pkt_sink = cncp_dst

    ft = generate_fib(ft, all_flows)

    for node_id in ft.nodes():
        node = ft.nodes[node_id]
        node["device"] = Router(env, nports=k, port_rate=rate_100, buffer_size=buffer_size,
                                routing_info=node["flow_to_port"], id=node_id, debug=True)
        node["wire"] = dict()
        for port_number, next_hop in node["port_to_nexthop"].items():
            node["wire"][port_number] = Wire(env, propagation_delay_dist)

    for node_id in ft.nodes():
        node = ft.nodes[node_id]
        if node["type"] is not "host":
            node["ratecontrolMonitor"] = RateControlMonitor(env, root_router=node["device"],
                                                            id=node_id, dist=delay_dist_monitor)
            downstream_hops = [ft.nodes[i]["device"] for i in node["nexthop"]]
            node["ratecontrolMonitor"].target_routers = downstream_hops

        if node["layer"] == "core" or node["layer"] == "aggregation":
            # add the bandwidth dynamics for the core and aggregation switches
            node["modify_bandwidth_monitor"] = modify_bandwidth_monitor(env, router=node["device"], random_seed=node_id,
                                                                        change_time=CHANGE_TIME, variance=variance, id=node_id,
                                                                        dist=delay_dist_monitor_dynamics)

    for n in ft.nodes():
        node = ft.nodes[n]
        for port_number, next_hop in node["port_to_nexthop"].items():
            node["device"].ports[port_number].out = node["wire"][port_number]
            node["wire"][port_number].out = ft.nodes[next_hop]["device"]

    for flow_id, flow in all_flows.items():
        flow.pkt_gen.out = ft.nodes[flow.src]["device"]
        ft.nodes[flow.dst]["device"].ends[flow_id] = flow.pkt_sink
        # ft.nodes[flow.dst]["ratecontrolMonitor"].target_routers = [flow.pkt_sink]

        ft.nodes[flow.src]["resetMonitor"] = ResetMonitor(env, source=flow.pkt_gen, routers=[ft.nodes[i]["device"] for i in flow.experi_intermediate_nodes],
                                     id=flow_id, dist=delay_dist_monitor_reset)

    output_log = output_Monitor(env=env, dist=delay_dist_output_monitor)

    env.run(until=finish_time)

    # record the file transmission information for all flows
    with open(f"./results/fct-CNCP-single-path-{test_id}.csv", "a", newline='', encoding="utf-8") as csvf1:
        writer1 = csv.writer(csvf1)
        writer1.writerow([f"finish time: {finish_time}, variance: {variance}, interval: {interval}, dynamic networks"])
        for flow_id in range(n_flows):
            fct = list()  # record the FCT for a flow
            for i in range(len(all_flows[flow_id].done_times)):
                x = all_flows[flow_id].actual_file_arrival_time[i]
                y = all_flows[flow_id].done_times[i]
                if y is not None:
                    fct.append(y - x)
            writer1.writerow(fct)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"total running of the simulation: {elapsed_time} s")








