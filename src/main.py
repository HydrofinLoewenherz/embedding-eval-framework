from dataclasses import dataclass
import numpy as np
import networkx as nx
import tensorflow as tf
import math
from tqdm.notebook import tqdm
import random

@dataclass
class Args:
    random_seed = None
    # tensorflow
    batch_size = 64
    epochs = 30
    layers = 10
    layer_size = 16
    train_size = 0.7
    wandb = False
    # graph gen
    graph_size = 1000
    graph_shape = 'disc'
    rg_radius = 0.05
    # dataset manipulation
    ds_padded = True


Node = (float, float)
Nodes = list[Node]
NodeIndexPairs = list[(int, int)]

Dataset = list(([float, float, float, float], float))


def gen_nodes(args: Args) -> Nodes:
    if args.graph_shape == 'disc':
        return __gen_nodes_disc(args.graph_size)
    else:
        raise f'unsupported node shape: {args.graph_shape}'


def __gen_nodes_disc(amount: int) -> Nodes:
    points = []
    with tqdm(total=amount, desc="generating random-uniform nodes on disc") as pbar:
        while len(points) < amount:
            p = (random.uniform(0, 1), random.uniform(0, 1))
            d = (p[0] - 0.5, p[1] - 0.5)
            if math.sqrt(d[0] * d[0] + d[1] * d[1]) > 0.5:
                continue
            points.append(p)
            pbar.update(1)
    return points


def get_node_pairs(n_nodes: int) -> NodeIndexPairs:
    return [
        (i0, i1)
        for i0 in tqdm(range(n_nodes), desc="generating node pairs")
        for i1 in range(i0 + 1, n_nodes)
    ]


# https://stackoverflow.com/a/36460020/10619052
def list_to_dict(items: list) -> dict:
    return {v: k for v, k in enumerate(tqdm(items, desc="creating dict from list"))}


def build_model(args: Args) -> tf.keras.Sequential:
    print('building model layers')
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=4),
        *[tf.keras.layers.Dense(args.layer_size, activation='relu') for _ in range(args.layers)],
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    print('compiling model')
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            'accuracy',
            tf.keras.metrics.Recall(thresholds=0),
            tf.keras.metrics.AUC(
                curve="PR"
            ),
        ]
    )
    return model


def nodes_to_dataset(graph: nx.Graph, nodes: Nodes, node_index_pairs : NodeIndexPairs) -> Dataset:
    return [
        (
            [*nodes[i0], *nodes[i1]],
            1 if graph.has_edge(i0, i1) else 0
        )
        for (i0, i1) in tqdm(node_index_pairs, desc="generating dataset from node pairs")
    ]


def run_model(model: tf.keras.Sequential, ds: Dataset, args: Args):
    # prepare dataset
    values, labels = zip(*ds)
    tf_values = tf.constant(np.array(values))
    tf_labels = tf.constant(np.array(labels))
    print("tessst")

    n_values = len(values)
    n_train = np.ceil(args.train_size * n_values)
    print(f'splitting dataset of {n_values} values into {n_train}/{n_values - n_train}')
    # prepare dataset
    #ds_full = tf.data.Dataset\
    #    .from_tensor_slices((tf_values, tf_labels))\
    #    .batch(args.batch_size)\
    #    .shuffle(1000)
    ds_full = tf.data.Dataset.from_tensor_slices((tf_values, tf_labels))
    print(f"len ds {ds_full.cardinality()}")
    ds_train = ds_full.take(n_train)
    ds_test = ds_full.skip(n_train)
    print(f"len ds {ds_full.cardinality()}")
    print(f"train: {ds_train.cardinality()}, test: {ds_test.cardinality()}")
    # fit & evaluate
    print('fitting model')
    model.fit(ds_train, epochs=args.epochs, verbose=1)
    print('evaluating model')
    return model.evaluate(ds_test, verbose=1)


def run(args: Args):
    nodes = gen_nodes(args)
    n_nodes = len(nodes)

    graph = nx.random_geometric_graph(
        args.graph_size,
        args.rg_radius,
        pos=list_to_dict(nodes)
    )

    node_index_pairs = get_node_pairs(n_nodes)
    dataset = nodes_to_dataset(graph, nodes, node_index_pairs)

    model = build_model(args)
    result = list(run_model(model, dataset, args))
    return result