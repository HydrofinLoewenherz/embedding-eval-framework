from dataclasses import dataclass


@dataclass(
    repr=True
)
class Args:
    random_seed = None
    # torch
    batch_size = 10
    epochs = 100
    layers = 10
    layer_size = 16
    train_size = 0.8
    eval_epochs = True
    # graph
    graph_size = 800
    graph_shape = 'disc'
    rg_radius = 0.05
    subsample = "bfs"
    subsample_size = 100
    subsample_ranking = "node_degree"
    balance = False
