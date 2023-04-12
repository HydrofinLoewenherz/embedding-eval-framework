import itertools
import math
import numpy as np
import random
import networkx as nx
from typing import Set, Tuple, Union, Dict, List
from girg_sampling import girgs
from copy import deepcopy

from src.args import Args

# Type alias to make the code more comprehensible
NodeData = Dict
NodeId = int
NodeWithData = Tuple[NodeId, NodeData]
NodeDataPairs = Set[Tuple[NodeWithData, NodeWithData]]
NodePos = Tuple[float, float]


node_shifts = {
    "none": (0, 0),
    "top": (1, 0),
    "down": (-1, 0),
    "left": (0, -1),
    "right": (0, 1),
    "top-left": (1, -1),
    "top-right": (1, 1),
    "bottom-left": (-1, -1),
    "bottom-right": (-1, 1),
}


def back_shift(key: str) -> str:
    (sx, sy) = node_shifts[key]
    return list(node_shifts.keys())[list(node_shifts.values()).index((-sx, -sy))]


def gen_graph(args: Args) -> nx.Graph:
    if args.graph_type == "rgg":
        radius = math.sqrt(args.rg_avg_degree / (args.graph_size * math.pi))
        graph = random_geometric_graph(
            size=args.graph_size,
            radius=radius
        )
    elif args.graph_type == "t_rgg":
        radius = math.sqrt(args.rg_avg_degree / (args.graph_size * math.pi))
        graph = toroid_random_geometric_graph(
            size=args.graph_size,
            radius=radius
        )
    elif args.graph_type == "random":
        radius = math.sqrt(args.rg_avg_degree / (args.graph_size * math.pi))
        graph = random_geometric_graph(
            size=args.graph_size,
            radius=radius
        )
        randomize_features(graph)
    elif args.graph_type == "girg":
        graph = girg_graph(
            size=args.graph_size
        )
    else:
        raise ValueError(f"invalid graph type: {args.graph_type}")
    return graph


def girg_graph(
        size: int,
        ple: float = 2.5,
        alpha: float = math.inf,
        deg: float = 10,
) -> nx.Graph:
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
    return graph


def gen_pos(disc: bool = True) -> NodePos:
    while True:
        p = (random.uniform(0, 1), random.uniform(0, 1))
        # if disc mode continue until pos in disc is generated
        d = (p[0] - 0.5, p[1] - 0.5)
        if disc and math.sqrt(d[0] * d[0] + d[1] * d[1]) > 0.5:
            continue
        return p


def gen_edge(d: float, disc: bool = True) -> Tuple[NodePos, NodePos]:
    while True:
        u, v = gen_pos(disc=disc), gen_pos(disc=disc)
        uv = tuple(np.subtract(v, u))
        w = np.add(u, (uv / np.linalg.norm(uv)) * d)
        # continue until 'w' is inside the area
        if 0 > w[0] > 1 or 0 > w[1] > 1:
            continue
        if disc and np.linalg.norm(w) > 1:
            continue
        return u, w


def random_geometric_graph(size: int, radius: float, disc: bool = True) -> nx.Graph:
    # generate graph from positions and radius
    node_positions = {i: gen_pos(disc=disc) for i in range(size)}
    graph = nx.random_geometric_graph(
        size,
        radius,
        pos=node_positions
    )
    nx.set_node_attributes(graph, node_positions, name="pos")
    nx.set_node_attributes(graph, node_positions, name="feature")
    return graph


def toroid_random_geometric_graph(size: int, radius: float) -> nx.Graph:
    graph = random_geometric_graph(size, radius, disc=False)
    # add additional edges
    graph.add_edges_from([
        (u, v, {"shift": key})
        for ((u, p_u), (v, (p_v_x, p_v_y))) in itertools.combinations(graph.nodes(data="pos"), 2)
        for (key, (s_x, s_y)) in node_shifts.items()
        if key != 'none' and np.abs(math.dist(p_u, (p_v_x + s_x, p_v_y + s_y))) <= radius
    ])
    return graph


def randomize_features(graph: nx.Graph):
    node_positions = {i: gen_pos() for i in list(graph.nodes)}
    nx.set_node_attributes(graph, node_positions, name="feature")


def sorted_nodes(graph: nx.Graph) -> nx.Graph:
    out = nx.Graph()
    out.add_nodes_from(
        sorted(list(graph.nodes(data=True)), key=lambda x: x[1]["feature"])
    )
    out.add_edges_from(graph.edges(data=True))
    return out


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


def bfs_subgraph(
        graph: nx.Graph,
        size: int
) -> Tuple[nx.Graph, Set[NodeId]]:
    visit: List[NodeId] = []
    visited: Set[NodeId] = set()
    while len(visited) < size:
        if len(visit) == 0:
            visit.append(random.choice(list(graph.nodes)))

        curr_node = visit.pop(0)
        neighbors = list(graph.neighbors(curr_node)) if curr_node is not None else []
        visit.extend([
            n
            for n in neighbors
            if n not in visited and n not in visit
        ])
        visited.add(curr_node)
    # build subgraph
    return graph.subgraph(visited), visited


def dfs_subgraph(
        graph: nx.Graph,
        size: int
) -> Tuple[nx.Graph, Set[NodeId]]:
    visit: List[NodeId] = []
    visited: Set[NodeId] = set()
    while len(visited) < size:
        if len(visit) == 0:
            visit.append(random.choice(list(graph.nodes)))

        curr_node = visit.pop(len(visit) - 1)
        neighbors = list(graph.neighbors(curr_node)) if curr_node is not None else []
        visit.extend([
            n
            for n in neighbors
            if n not in visited and n not in visit
        ])
        visited.add(curr_node)
    # build subgraph
    return graph.subgraph(visited), visited


def on_toriod(graph: nx.Graph) -> nx.Graph:
    # create a copy of the graph to ensure that the original graph is not altered for the representation
    graph = deepcopy(graph)
    t_graph = nx.Graph()

    # add shifted copies for all nodes
    # the original node key is also replaced with the shift suffix 'none'
    t_graph.add_nodes_from([
        (
            f"{u}_{key}",
            {
                **d,
                "pos": [d["pos"][0] + xs, d["pos"][1] + ys],
                "shift": key,
                "original": u
            }
        )
        for (u, d) in graph.nodes(data=True)
        for (key, (xs, ys)) in node_shifts.items()
    ])

    # add one edge for each original edge
    # for each possible shift, use the edge that is the shortest
    positions = t_graph.nodes.data("pos")
    shortest_edges = [
        sorted([
            (u, v, {
                **d,
                "shift": key,
                "dist": math.dist(
                    positions[f"{u}_none"],
                    (positions[f"{v}_none"][0] + sx, positions[f"{v}_none"][1] + sy)
                )
            })
            for [key, (sx, sy)] in node_shifts.items()
        ], key=lambda x: x[2]["dist"])[0]
        for (u, v, d) in graph.edges(data=True)
    ]

    # for each shifted edge, add two halve edges u -> v' and u' -> v
    # un-shifted edges are added twice, but that should not matter
    t_graph.add_edges_from([
        (
            f"{u}_none",
            f"{v}_{d['shift']}",
            d
        )
        if lr else
        (
            f"{v}_none",
            f"{u}_{back_shift(d['shift'])}",
            d
        )
        for (u, v, d) in shortest_edges
        for lr in [True, False]
    ])
    return t_graph


def toroid_edge_shift(node: str) -> str:
    return node.split("_", 1)[0]
