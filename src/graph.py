import math
import random
import networkx as nx
from typing import Set, Tuple, Union, Dict

# Type alias to make the code more comprehensible
NodeData = Dict
NodeId = int
NodeWithData = Tuple[NodeId, NodeData]
NodeDataPairs = Set[Tuple[NodeWithData, NodeWithData]]


def random_geometric_graph(size: int, radius: float) -> nx.Graph:
    # generate positions on unit-disc
    positions = []
    while len(positions) < size:
        p = (random.uniform(0, 1), random.uniform(0, 1))
        d = (p[0] - 0.5, p[1] - 0.5)
        if math.sqrt(d[0] * d[0] + d[1] * d[1]) > 0.5:
            continue
        positions.append(p)
    # generate graph from positions and radius
    node_positions = {i: p for i, p in enumerate(positions)}
    graph = nx.random_geometric_graph(
        size,
        radius,
        pos=node_positions
    )
    nx.set_node_attributes(graph, node_positions, name="pos")
    return graph


def subgraph(graph: nx.Graph, size: int, alpha: float = 1.0, boredom_pth: float = 0.3) -> nx.Graph:
    curr: Union[NodeId, None] = None
    sampled: Set[NodeId] = set()
    boredom: int = 0  # escape parameter in case it walks on a too small component, enforces jumps
    while len(sampled) < size:
        neighbors = list(graph.neighbors(curr)) if curr is not None else []
        # random jump (first iteration is always a jump)
        if len(neighbors) == 0 \
                or alpha == 1 \
                or random.random() >= alpha \
                or (boredom > size * boredom_pth):
            curr = random.choice(list(graph.nodes))
        # random walk
        else:
            curr = random.choice(neighbors)
        # add found node if new, otherwise increase boredom
        if curr in sampled:
            boredom = boredom + 1
            continue
        boredom = 0
        sampled.add(curr)
    # build subgraph
    return graph.subgraph(sampled)
