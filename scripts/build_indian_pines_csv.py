from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.io import loadmat


def build_csv(
    data_mat: Path,
    gt_mat: Path,
    output_csv: Path,
    labels_output_csv: Path | None = None,
) -> None:
    data = loadmat(data_mat)["indian_pines_corrected"]
    labels = loadmat(gt_mat)["indian_pines_gt"]

    flat_data = data.reshape(-1, data.shape[2])
    df = pd.DataFrame(flat_data)
    df = pd.concat([df, pd.DataFrame(labels.ravel())], axis=1)
    df.columns = [f"band{i}" for i in range(1, data.shape[2] + 1)] + ["class"]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    if labels_output_csv is not None:
        labels_output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(labels).to_csv(labels_output_csv, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a flattened Indian Pines CSV from .mat files."
    )
    parser.add_argument(
        "--data-mat",
        type=Path,
        default=Path("dataset/Indian_pines_corrected.mat"),
        help="Path to Indian_pines_corrected.mat",
    )
    parser.add_argument(
        "--gt-mat",
        type=Path,
        default=Path("dataset/Indian_pines_gt.mat"),
        help="Path to Indian_pines_gt.mat",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("indian_pines_all.csv"),
        help="Output CSV path for flattened samples",
    )
    parser.add_argument(
        "--labels-output",
        type=Path,
        default=Path("df_indian_pines_gt.csv"),
        help="Optional label CSV output path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_csv(
        data_mat=args.data_mat,
        gt_mat=args.gt_mat,
        output_csv=args.output,
        labels_output_csv=args.labels_output,
    )
    print(f"Saved sample CSV to: {args.output}")
    print(f"Saved label CSV to: {args.labels_output}")


if __name__ == "__main__":
    main()
