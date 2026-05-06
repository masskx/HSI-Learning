"""Inference, metrics and artifact export helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .utils import ensure_dir, save_json


@torch.no_grad()
def predict_full_image(model: torch.nn.Module, loader, image_shape: tuple[int, int], device: torch.device):
    """Run full-image inference patch by patch and rebuild a prediction map."""
    model.eval()
    predictions = []
    for batch in loader:
        inputs = batch.to(device)
        logits = model(inputs)
        predictions.append(logits.argmax(dim=1).cpu().numpy() + 1)

    flat_map = np.hstack(predictions).astype(np.uint8)
    return flat_map.reshape(image_shape)


def compute_classification_metrics(test_gt: np.ndarray, pred_map: np.ndarray, class_names: list[str]):
    """Compute OA, AA, kappa and a per-class report on the held-out pixels."""
    from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, recall_score

    test_pred = pred_map[test_gt != 0]
    test_true = test_gt[test_gt != 0]
    labels = list(range(1, len(class_names) + 1))

    oa = accuracy_score(test_true, test_pred)
    aa = recall_score(test_true, test_pred, labels=labels, average="macro", zero_division=0)
    kappa = cohen_kappa_score(test_true, test_pred, labels=labels)
    report = classification_report(
        test_true,
        test_pred,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    report_text = f"OA: {oa:.6f}\nAA: {aa:.6f}\nKappa: {kappa:.6f}\n\n{report}"
    return {"oa": oa, "aa": aa, "kappa": kappa, "report_text": report_text}


def save_classification_report(report_text: str, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    output_path.write_text(report_text, encoding="utf-8")
    return output_path


def save_prediction_artifacts(
    pred_map: np.ndarray,
    gt: np.ndarray,
    output_dir: str | Path,
    prefix: str = "prediction",
):
    """Save raw arrays and RGB visualizations for prediction maps."""
    output_dir = ensure_dir(output_dir)
    masked_map = pred_map * (gt != 0)
    preview_path = output_dir / f"{prefix}.jpg"
    masked_preview_path = output_dir / f"{prefix}_masked.jpg"

    np.save(output_dir / f"{prefix}.npy", pred_map)
    np.save(output_dir / f"{prefix}_masked.npy", masked_map)

    preview_available = False
    preview_error = None
    try:
        import spectral

        spectral.save_rgb(preview_path, pred_map, colors=spectral.spy_colors)
        spectral.save_rgb(masked_preview_path, masked_map, colors=spectral.spy_colors)
        preview_available = True
    except ImportError as exc:
        preview_error = str(exc)

    summary = {
        "prediction_path": output_dir / f"{prefix}.npy",
        "masked_prediction_path": output_dir / f"{prefix}_masked.npy",
        "preview_available": preview_available,
        "preview_path": preview_path if preview_available else None,
        "masked_preview_path": masked_preview_path if preview_available else None,
        "preview_error": preview_error,
    }
    save_json(summary, output_dir / f"{prefix}_artifacts.json")
    return summary
