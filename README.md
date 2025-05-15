# Python-based Network Simulator for Coding-based Non-conservative Communication Protocol (CNCP)
This is a Github repository for the INFOCOM'25 paper "Erasure Coding-based Non-conservative Network Communication: A Ground Up Approach"

This network simulator is based on [ns.py](https://github.com/TL-System/ns.py), which is a pythonic discrete-event network simulator. 

## RUN
First, install the required libraries:
```
$ conda install -f cncp.yaml
```
Then, activate the conda environment:
```
$ conda activate cncp
```

## NOTE
We briefly introduce each file in this repository:

- `cncp_v2.py`: define and implement the basic components for CNCP, including the source node (`DistPacketGenerator` class), the intermediate node (`Router` class), the destination node (`PacketSink` class), and the rate update module (`RateControlMonitor` class and `ResetMonitor` class). Moreover, this file includes a tot-example to verify the effectiveness of CNCP.
- `fattree_cncp_single_path.py`: give the general network setting to test CNCP in the fat-tree topology, including constructing a fat-tree topology, Gaussian distribution-based network dynamics, recording the testing results for CNCP, etc.. More details can be found in our paper.
- `src_dest_pair.txt`: include randomly generated source-destination pairs.
- `flow_path.txt`: include the path for each source-destination pair.
