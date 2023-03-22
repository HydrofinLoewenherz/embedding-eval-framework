import math
import random
import networkx as nx
from typing import Set, Tuple, Union, Dict
from girg_sampling import girgs

# Type alias to make the code more comprehensible
NodeData = Dict
NodeId = int
NodeWithData = Tuple[NodeId, NodeData]
NodeDataPairs = Set[Tuple[NodeWithData, NodeWithData]]


def girg_graph(
        size: int,
        ple: float = 2.5,
        alpha: float = math.inf,
        deg: float = 10,
) -> Tuple[nx.Graph, int]:
    # generate girg
    weights = girgs.generateWeights(size, ple)
    positions = girgs.generatePositions(size, 2)
    weights_scale = girgs.scaleWeights(weights, deg, 2, alpha)
    scaled_weights = [x * weights_scale for x in weights]
    edges = girgs.generateEdges(scaled_weights, positions, alpha)

    # convert to NetworkX graph (use pos and weight as feature)
    node_positions = {i: pos for i, pos in enumerate(positions)}
    node_features = {
        i: (*positions[i], scaled_weights[i])
        for i in range(size)
    }
    graph = nx.Graph()
    graph.add_nodes_from(range(size))
    nx.set_node_attributes(graph, node_positions, name="pos")
    nx.set_node_attributes(graph, node_features, name="feature")
    graph.add_edges_from(edges)
    return graph, 3


def gen_disc_pos() -> Tuple[float, float]:
    while True:
        p = (random.uniform(0, 1), random.uniform(0, 1))
        d = (p[0] - 0.5, p[1] - 0.5)
        if math.sqrt(d[0] * d[0] + d[1] * d[1]) > 0.5:
            continue
        return p


def random_geometric_graph(size: int, radius: float) -> Tuple[nx.Graph, int]:
    # generate graph from positions and radius
    node_positions = {i: gen_disc_pos() for i in range(size)}
    graph = nx.random_geometric_graph(
        size,
        radius,
        pos=node_positions
    )
    nx.set_node_attributes(graph, node_positions, name="pos")
    nx.set_node_attributes(graph, node_positions, name="feature")
    return graph, 2


def random_graph(size: int) -> Tuple[nx.Graph, int]:
    node_positions = {i: gen_disc_pos() for i in range(size)}
    # generate edges (the specific alg used is not of interest)
    graph = nx.powerlaw_cluster_graph(
        n=size,
        m=int(size / 2),
        p=0.5
    )
    # give nodes random positions and features
    # even if the graph is a powerlaw-cluster-graph, as long as the embedding is random, it shouldn't give a good score
    nx.set_node_attributes(graph, node_positions, name="pos")
    nx.set_node_attributes(graph, node_positions, name="feature")
    return graph, 2


def subgraph(
        graph: nx.Graph,
        size: int,
        alpha: float = 1.0,
        boredom_pth: float = 0.3,
        ignore=None
) -> Tuple[nx.Graph, Set[NodeId]]:
    # init
    curr_node: Union[NodeId, None] = None
    sampled: Set[NodeId] = set()
    boredom: int = 0  # escape parameter in case it walks on a too small component, enforces jumps
    while len(sampled) < size:
        neighbors = list(graph.neighbors(curr_node)) if curr_node is not None else []
        # random jump (first iteration is always a jump)
        if len(neighbors) == 0 \
                or random.random() < alpha \
                or (boredom > size * boredom_pth):
            next_node = random.choice(list(graph.nodes))
        # random walk
        else:
            next_node = random.choice(neighbors)
        # skip ignore list (don't even allow walking over it)
        if ignore is not None and next_node in ignore:
            boredom = boredom + 1
            continue
        curr_node = next_node
        # add found node if new, otherwise increase boredom
        if curr_node in sampled:
            boredom = boredom + 1
            continue
        boredom = 0
        sampled.add(curr_node)
    # build subgraph
    return graph.subgraph(sampled), sampled
