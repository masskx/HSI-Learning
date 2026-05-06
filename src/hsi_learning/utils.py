"""Shared helpers for scripts and training modules."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def ensure_dir(path: str | Path) -> Path:
    """Create a directory when needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | int) -> torch.device:
    """Resolve a user-friendly device selector into a torch.device."""
    if isinstance(device, int):
        if device < 0:
            return torch.device("cpu")
        if torch.cuda.is_available():
            return torch.device(f"cuda:{device}")
        return torch.device("cpu")

    text = str(device).strip().lower()
    if text in {"auto", ""}:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if text == "cpu":
        return torch.device("cpu")
    if text == "cuda" and torch.cuda.is_available():
        return torch.device("cuda:0")
    if text.startswith("cuda:") and torch.cuda.is_available():
        return torch.device(text)
    return torch.device("cpu")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def save_json(payload: Any, path: str | Path) -> Path:
    """Write JSON with stable formatting."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return output_path
