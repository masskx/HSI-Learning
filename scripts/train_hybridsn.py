"""Train and evaluate HybridSN outside the notebook."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train HybridSN on a hyperspectral dataset.")
    parser.add_argument("--dataset", default="IP", choices=["IP", "SA", "PU"], help="Dataset code.")
    parser.add_argument("--dataset-dir", default="dataset", help="Directory containing the .mat files.")
    parser.add_argument("--train-rate", type=float, default=0.3, help="Training split ratio.")
    parser.add_argument("--val-rate", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--eval-interval", type=int, default=5, help="Validation interval in epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-6, help="Weight decay.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--pca-components", type=int, default=15, help="Number of PCA components.")
    parser.add_argument("--patch-size", type=int, default=25, help="Spatial patch size.")
    parser.add_argument("--seed", type=int, default=63466, help="Random seed.")
    parser.add_argument("--device", default="auto", help="cpu, cuda, cuda:0, auto, or an integer index.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader worker count.")
    parser.add_argument(
        "--output-dir",
        default="results/hybridsn",
        help="Base output directory. Dataset code is appended automatically.",
    )
    parser.add_argument(
        "--keep-all-checkpoints",
        action="store_true",
        help="Keep every validation checkpoint instead of only the best one.",
    )
    return parser


def main() -> int:
    import torch

    from hsi_learning.data import apply_pca_cube, build_sample_report, create_dataloaders, load_hsi_dataset
    from hsi_learning.engine import TrainingConfig, fit
    from hsi_learning.evaluation import (
        compute_classification_metrics,
        predict_full_image,
        save_classification_report,
        save_prediction_artifacts,
    )
    from hsi_learning.models import HybridSN
    from hsi_learning.utils import ensure_dir, resolve_device, save_json, set_seed

    parser = build_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(Path(args.output_dir) / args.dataset)

    cube, gt, class_names = load_hsi_dataset(args.dataset, dataset_dir=args.dataset_dir)
    reduced_cube, pca = apply_pca_cube(cube, n_components=args.pca_components)

    dataloaders = create_dataloaders(
        reduced_cube,
        gt,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        train_rate=args.train_rate,
        val_rate=args.val_rate,
        random_state=args.seed,
        num_workers=args.num_workers,
    )

    sample_report = build_sample_report(
        gt,
        dataloaders["train_gt"],
        dataloaders["val_gt"],
        dataloaders["test_gt"],
        class_names=class_names,
    )
    print(sample_report)
    (output_dir / "sample_report.txt").write_text(sample_report, encoding="utf-8")

    config_payload = {
        "dataset": args.dataset,
        "dataset_dir": args.dataset_dir,
        "train_rate": args.train_rate,
        "val_rate": args.val_rate,
        "epochs": args.epochs,
        "eval_interval": args.eval_interval,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "pca_components": args.pca_components,
        "patch_size": args.patch_size,
        "seed": args.seed,
        "device": str(device),
        "cube_shape": list(cube.shape),
        "reduced_cube_shape": list(reduced_cube.shape),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }
    save_json(config_payload, output_dir / "run_config.json")

    model = HybridSN(
        input_channels=args.pca_components,
        patch_size=args.patch_size,
        num_classes=len(class_names),
    ).to(device)

    fit_result = fit(
        model=model,
        train_loader=dataloaders["train_loader"],
        val_loader=dataloaders["val_loader"],
        device=device,
        config=TrainingConfig(
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            eval_interval=args.eval_interval,
            output_dir=output_dir,
            keep_all_checkpoints=args.keep_all_checkpoints,
        ),
    )

    model.load_state_dict(torch.load(fit_result.best_checkpoint, map_location=device))
    pred_map = predict_full_image(
        model,
        dataloaders["pred_loader"],
        image_shape=gt.shape,
        device=device,
    )
    save_prediction_artifacts(pred_map, gt, output_dir)

    metrics = compute_classification_metrics(dataloaders["test_gt"], pred_map, class_names)
    save_classification_report(metrics["report_text"], output_dir / "classification_report.txt")
    save_json(
        {
            "oa": metrics["oa"],
            "aa": metrics["aa"],
            "kappa": metrics["kappa"],
            "best_checkpoint": fit_result.best_checkpoint,
            "best_val_acc": fit_result.best_val_acc,
        },
        output_dir / "metrics.json",
    )

    print(f"Best checkpoint: {fit_result.best_checkpoint}")
    print(f"Validation accuracy: {fit_result.best_val_acc:.4f}")
    print(metrics["report_text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
