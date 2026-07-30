"""CRC-GAD experiment configuration — matches paper Section 5.5."""
from dataclasses import dataclass, field
from typing import List


SEEDS: List[int] = [0, 1, 2, 3, 4]
DATASETS: List[str] = ["cora", "citeseer", "pubmed", "acm", "blogcatalog", "flickr"]
ALPHAS: List[float] = [0.01, 0.02, 0.05, 0.10]
CONTAMINATION_LEVELS: List[float] = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]
REWIRE_LEVELS: List[float] = [0.0, 0.10, 0.20, 0.40, 0.60]
N_PARTITION_SPLITS: int = 200  # dependence study


@dataclass
class Config:
    # Node partition (Section 5.5)
    train_frac: float = 0.60
    val_frac_of_remain: float = 0.10
    rho: float = 0.30  # |C| / |V_eval|

    # CoLA-style scorer (Liu et al. 2022) — Section 5.5
    embed_dim: int = 128
    subgraph_size: int = 8
    num_negatives: int = 5  # M
    gcn_layers: int = 2
    rwr_restart: float = 0.15
    scoring_rounds: int = 32  # R (use 256 for full paper runs)
    val_scoring_rounds: int = 8  # R_val
    batch_size: int = 256
    lr: float = 0.001
    max_epochs: int = 120
    val_check_every: int = 10  # E_val
    patience: int = 10

    # Anomaly injection (DOMINANT / CoLA protocol)
    anomaly_ratio: float = 0.05
    structural_ratio: float = 0.5  # half structural, half attribute

    # Conformal — labels NEVER used for C/T construction
    use_label_clean_calibration: bool = False

    seeds: List[int] = field(default_factory=lambda: SEEDS.copy())

    def flagged_rate_upper_bound(self, alpha: float, n_cal: int) -> float:
        """Theorem 2: max marginal flagged rate on mixed test set."""
        import math
        return math.floor(alpha * (n_cal + 1)) / (n_cal + 1)


DEFAULT = Config()
