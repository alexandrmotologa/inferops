"""Unit tests for GPU Interconnect Topology discovery and analysis."""

from inferops.hardware.topology import parse_topology_matrix


def test_parse_topology_matrix():
    # Sample matrix output from nvidia-smi topo -m on an 2x A100 system
    sample_output = """
        GPU0	GPU1	CPU Affinity	NUMA Affinity	GPU NUMA ID
GPU0	 X 	NV12	0-63	0	N/A
GPU1	NV12	 X 	0-63	0	N/A

Legend:
  X    = Self
  SYS  = Connection traversing PCIe as well as interconnect between NUMA nodes
  NV#  = Connection traversing a bonded set of # NVLinks
"""
    links = parse_topology_matrix(sample_output)
    assert len(links) == 1
    assert links[0].gpu_a == 0
    assert links[0].gpu_b == 1
    assert links[0].is_nvlink is True
    assert links[0].is_high_speed is True
    assert links[0].link_type == "NV12"


def test_parse_topology_matrix_pcie_sys():
    # Sample matrix output from a system with slow inter-socket connection
    sample_output = """
        GPU0	GPU1	CPU Affinity	NUMA Affinity	GPU NUMA ID
GPU0	 X 	SYS	0-31	0	N/A
GPU1	SYS	 X 	32-63	1	N/A
"""
    links = parse_topology_matrix(sample_output)
    assert len(links) == 1
    assert links[0].gpu_a == 0
    assert links[0].gpu_b == 1
    assert links[0].is_nvlink is False
    assert links[0].is_high_speed is False
    assert links[0].link_type == "SYS"
