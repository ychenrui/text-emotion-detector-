"""Small helpers shared across the pipeline."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def get_device() -> torch.device:
    """Pick the best torch device available.

    Targets CUDA first (RTX 4090). Falls back to MPS for Apple Silicon and
    finally CPU so the same code still runs on a laptop for sanity checks.
    On CUDA we also enable TF32 — a small free win on Ampere/Ada in fp32
    paths (it's a no-op while we're running bf16 mixed precision, but keeps
    eval/inference fast).
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
