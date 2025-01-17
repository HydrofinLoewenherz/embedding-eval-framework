import math
from dataclasses import dataclass, field

# TODO update all datasets to updated "dataset_split" format
# result_drop = result.drop(labels=["dataset_split"], axis=1)
# result_drop["dataset_split"] = "60-20-20"
# result_drop


@dataclass(
    repr=True,
    unsafe_hash=True
)
class Args:
    # torch settings
    epochs: int = field(default=250)
    batch_size: int = field(default=64)
    sort_dataset: bool = field(default=False)
    std_dataset: bool = field(default=True)
    dataset_split: str = field(default="60-20-20")  # [train]-[valid]-[test]
    early_stopping: bool = field(default=True)
    threshold_stopping: bool = field(default=True)
    # graph settings
    graph_type: str = field(default="rgg")
    graph_size: int = field(default=1000)
    # graph type specific settings
    rgg_avg_degree: int = field(default=10)
    girg_ple: float = field(default=2.5)
    girg_alpha: float = field(default=math.inf)
    girg_deg: float = field(default=10)
    # subgraph settings
    subgraph_alg: str = field(default="rjs")
    subgraph_size: int = field(default=100)
    # subgraph type specific settings
    subgraph_alpha: float = field(default=0.0)
    subgraph_beta: float = field(default=0.8)
