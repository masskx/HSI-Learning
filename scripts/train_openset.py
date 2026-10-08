"""Corrected open-set lecture entry point.

Known-only training/validation; frozen calibration thresholds; explicit
unknown IDs (not a hidden class-frequency heuristic). Both scores use higher
= more unknown. Saves model, preprocessing, split, thresholds and metrics.

    python scripts/train_openset.py --epochs 20
    python scripts/train_openset.py --unknown-classes 1 7 --output-dir results/openset_two

No test labels are used to select a model or a rejection threshold.
Same-scene spatial context may still contain unknown land cover.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from hsi_learning.teaching_runs import main

if __name__ == '__main__':
    main('openset')
