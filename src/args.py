import math
from dataclasses import dataclass, field
from typing import List


@dataclass(
    repr=True,
    unsafe_hash=True
)
class Args:  # TODO use typed dict?
    # torch
    batch_size: int = field(default=128)
    epochs: int = field(default=250)
    layers: int = field(default=10)
    layer_size: int = field(default=32)  # TODO relative to input size?
    sort_dataset: bool = field(default=True)
    test_split: float = field(default=0.2)
    valid_split: float = field(default=0.2)
    eval_epochs: bool = field(default=True)
    early_stopping: bool = field(default=True)
    # torch epoch graph
    epoch_graph_size: int = field(default=100)
    epoch_graph_alpha: float = field(default=0.0)
    epoch_graph_boredom_pth: float = field(default=0.8)
    # graph
    random_seed: any = field(default=None)
    graph_size: int = field(default=1000)
    graph_type: str = field(default="rgg")
    # random geometric graph
    rg_avg_degree: int = field(default=10)
    # girg graph
    g_ple: float = field(default=2.5)
    g_alpha: float = field(default=math.inf)
    g_deg: float = field(default=10)
    # note for cvs saving
    note: str = field(default="-")


def gridsearch_args() -> List[Args]:
    return [
        Args(
            graph_size=graph_size,
            epoch_graph_size=epoch_graph_size,
            epoch_graph_alpha=epoch_graph_alpha
        )
        # for graph_size in [5000]
        for graph_size in [500, 1000, 2500]
        for epoch_graph_size in [50, 100, 250]
        for epoch_graph_alpha in [0.0, 0.5, 1.0]
    ]
