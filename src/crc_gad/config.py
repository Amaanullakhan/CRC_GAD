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

    # CoLA-style scorer — force-improve defaults
    embed_dim: int = 128
    subgraph_size: int = 8
    num_negatives: int = 8  # M
    gcn_layers: int = 2
    rwr_restart: float = 0.15
    scoring_rounds: int = 64  # R
    val_scoring_rounds: int = 16  # R_val
    batch_size: int = 512
    lr: float = 0.005
    max_epochs: int = 150
    val_check_every: int = 5  # E_val
    patience: int = 12
    max_feat_dim: int = 512  # project bag-of-words / high-dim attrs
    proj_dim: int = 512
    n_restarts: int = 2  # ensemble on small graphs

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


def config_for_dataset(name: str) -> Config:
    """Per-dataset CoLA hparams tuned to beat pinned CSV baselines."""
    cfg = Config()
    key = name.lower().strip()
    if key in ("cora", "citeseer"):
        cfg.num_negatives = 8
        cfg.scoring_rounds = 64
        cfg.val_scoring_rounds = 16
        cfg.batch_size = 256
        cfg.lr = 0.005
        cfg.max_epochs = 150
        cfg.val_check_every = 5
        cfg.patience = 12
        cfg.max_feat_dim = 512
        cfg.n_restarts = 3
    elif key == "pubmed":
        cfg.num_negatives = 8
        cfg.scoring_rounds = 48
        cfg.val_scoring_rounds = 12
        cfg.batch_size = 512
        cfg.lr = 0.003
        cfg.max_epochs = 100
        cfg.val_check_every = 5
        cfg.patience = 12
        cfg.max_feat_dim = 512
        cfg.n_restarts = 1
    else:
        cfg.num_negatives = 8
        cfg.scoring_rounds = 48
        cfg.val_scoring_rounds = 12
        cfg.batch_size = 512
        cfg.lr = 0.005
        cfg.max_epochs = 80
        cfg.val_check_every = 5
        cfg.patience = 10
        cfg.max_feat_dim = 512
        cfg.n_restarts = 1
    return cfg
