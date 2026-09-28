"""Aggregate the Chapter-12 ablation study (3 seeds x 3 configs).

Reads results/ch12/<config>_seed<N>/metrics.json for
    configs: sir0 (lambda=0), sir01 (lambda=0.1), nores (no residual)
    seeds:   1334, 1335, 1336
computes mean +/- std for OA/AA/Kappa, and prints a markdown-ready table
plus a JSON summary saved to results/ch12_ablation_summary.json.

Usage:
    python scripts/aggregate_ch12_ablation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "ch12"

CONFIGS = [
    ("sir0", "SIR λ=0（基线）"),
    ("sir01", "SIR λ=0.1"),
    ("nores", "去残差（顺序堆叠）"),
]
SEEDS = [1334, 1335, 1336]


def main() -> int:
    summary = {}
    for tag, label in CONFIGS:
        rows = []
        for seed in SEEDS:
            # train_ssrn.py appends the dataset code: results/ch12/<tag>_seed<N>/IP/
            path = RESULTS / f"{tag}_seed{seed}" / "IP" / "metrics.json"
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        summary[label] = {
            "seeds": SEEDS,
            "oa": [r["oa"] for r in rows],
            "aa": [r["aa"] for r in rows],
            "kappa": [r["kappa"] for r in rows],
            "best_epoch": [r["best_epoch"] for r in rows],
            "oa_mean": float(np.mean([r["oa"] for r in rows])),
            "oa_std": float(np.std([r["oa"] for r in rows], ddof=1)),
            "aa_mean": float(np.mean([r["aa"] for r in rows])),
            "aa_std": float(np.std([r["aa"] for r in rows], ddof=1)),
            "kappa_mean": float(np.mean([r["kappa"] for r in rows])),
            "kappa_std": float(np.std([r["kappa"] for r in rows], ddof=1)),
        }

    out = RESULTS.parent / "ch12_ablation_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("| 配置 | OA (mean±std) | AA (mean±std) | Kappa (mean±std) | best epoch |")
    print("|---|---|---|---|---|")
    for label, s in summary.items():
        print(
            f"| {label} "
            f"| {s['oa_mean'] * 100:.2f} ± {s['oa_std'] * 100:.2f} "
            f"| {s['aa_mean'] * 100:.2f} ± {s['aa_std'] * 100:.2f} "
            f"| {s['kappa_mean']:.4f} ± {s['kappa_std']:.4f} "
            f"| {s['best_epoch']} |"
        )
    per_seed = {label: [round(r["oa"] * 100, 2) for r in
                        [dict(oa=v) for v in s["oa"]]]
                for label, s in summary.items()}
    print("\nper-seed OA:", json.dumps(per_seed, ensure_ascii=False))
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
