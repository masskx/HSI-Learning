"""Corrected lecture ProtoNet entry point.

Default: class-disjoint meta-train/val/test, train-only PCA/scaler,
negative squared Euclidean distance, validation-only checkpoint selection.
Historical results/protonet is not overwritten.

    python scripts/train_protonet.py --episodes 200 --eval-episodes 30
    python scripts/train_protonet.py --mode same-class --output-dir results/teaching_sameclass

K=1/5/10 are evaluated with the SAME frozen encoder and paired query tasks.
This replaces the old per-K training/overwritten-checkpoint workflow.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from hsi_learning.teaching_runs import main

if __name__ == '__main__':
    main('protonet')
