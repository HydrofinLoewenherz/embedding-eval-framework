import random
import networkx as nx
from copy import deepcopy


def sub_of(
        alg: str,
        graph: nx.Graph,
        size: int,
        alpha: float = 0.15,
        beta: float = 0.85,
        dc: bool = False
) -> nx.Graph:
    if alg == "wrs":
        return wrs_of(
            graph=graph,
            size=size,
            dc=dc
        )
    if alg == "bfs":
        return bfs_of(
            graph=graph,
            size=size,
            dc=dc
        )
    if alg == "dfs":
        return dfs_of(
            graph=graph,
            size=size,
            dc=dc
        )
    if alg == "rw":
        return rw_of(
            graph=graph,
            size=size,
            alpha=alpha,
            dc=dc
        )
    if alg == "rj":
        return rj_of(
            graph=graph,
            size=size,
            alpha=alpha,
            dc=dc
        )
    if alg == "rjs":
        return rjs_of(
            graph=graph,
            size=size,
            alpha=alpha,
            beta=beta,
            dc=dc
        )
    # if type does not match any known subgraph algorithm, raise error
    raise ValueError(f"unknown subgraph alg: {alg}")


def wrs_of(
        graph: nx.Graph,
        size: int,
        dc: bool = False
) -> nx.Graph:
    if dc:
        graph = deepcopy(graph)
    nodes = random.sample(
        list(graph.nodes),
        k=size
    )
    return nx.subgraph(graph, nodes)


def bfs_of(
        graph: nx.Graph,
        size: int,
        dc: bool = False
) -> nx.Graph:
    if dc:
        graph = deepcopy(graph)
    start = random.choice(list(graph.nodes))
    tree = nx.bfs_tree(graph, start, depth_limit=size)
    # the nodes were added in order of traversal
    nodes = list(tree.nodes())[:size]
    print(start, nodes)
    if len(nodes) != size:
        print(f"dfs only reached sample size {len(nodes)}/{size}")
    return nx.subgraph(graph, nodes)


def dfs_of(
        graph: nx.Graph,
        size: int,
        dc: bool = False
) -> nx.Graph:
    if dc:
        graph = deepcopy(graph)
    start = random.choice(list(graph.nodes))
    tree = nx.dfs_tree(graph, start, depth_limit=size)
    # the nodes were added in order of traversal
    nodes = list(tree.nodes())[:size]
    print(start, nodes)
    if len(nodes) != size:
        print(f"dfs only reached sample size {len(nodes)}/{size}")
    return nx.subgraph(graph, nodes)


def rw_of(
        graph: nx.Graph,
        size: int,
        alpha: float = 0.15,
        dc: bool = False
) -> nx.Graph:
    if dc:
        graph = deepcopy(graph)
    start = random.choice(list(graph.nodes))
    if len(list(graph.edges(start))) == 0:
        print(f"rw start node has no edges")
        return nx.Graph()
    curr = start
    nodes = {curr}
    # walk until enough nodes have been visited
    steps = 0
    while len(nodes) < size:
        if random.random() < alpha:  # do a jump back to the start
            curr = start
        else:  # do a random walk
            edges = list(graph.edges(curr))
            _, curr = random.choice(edges)
        nodes.add(curr)
        # break if too many steps
        steps += 1
        if steps > 100 * size:
            print(f"rw only reached sample size {len(nodes)}/{size} in {steps} steps")
            break
    return nx.subgraph(graph, nodes)


def rj_of(
        graph: nx.Graph,
        size: int,
        alpha: float = 0.15,
        dc: bool = False
) -> nx.Graph:
    if dc:
        graph = deepcopy(graph)
    start = random.choice(list(graph.nodes))
    curr = start
    nodes = {curr}
    # walk until enough nodes have been visited
    while len(nodes) < size:
        edges = list(graph.edges(curr))
        if random.random() < alpha or len(edges) == 0:  # do a random jump
            curr = random.choice(list(graph.nodes))
        else:  # do a random walk
            _, curr = random.choice(edges)
        nodes.add(curr)
    return nx.subgraph(graph, nodes)


def rjs_of(
        graph: nx.Graph,
        size: int,
        alpha: float = 0.15,
        beta: float = 0.85,
        dc: bool = False
) -> nx.Graph:
    if dc:
        graph = deepcopy(graph)
    start = random.choice(list(graph.nodes))
    curr = start
    nodes = {curr}
    # walk until enough nodes have been visited
    boredom = 0
    while len(nodes) < size:
        edges = list(graph.edges(curr))
        if random.random() < alpha or boredom > beta * size or len(edges) == 0:  # do a random jump
            curr = random.choice(list(graph.nodes))
        else:  # do a random walk
            _, curr = random.choice(edges)

        # update boredom counter
        if curr in nodes:
            boredom += 1
        else:
            boredom = 0

        nodes.add(curr)
    return nx.subgraph(graph, nodes)
