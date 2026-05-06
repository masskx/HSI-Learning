"""Dataset loading, splitting and patch extraction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class DatasetSpec:
    code: str
    data_file: str
    data_key: str
    gt_file: str
    gt_key: str
    class_names: tuple[str, ...]


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "IP": DatasetSpec(
        code="IP",
        data_file="Indian_pines_corrected.mat",
        data_key="indian_pines_corrected",
        gt_file="Indian_pines_gt.mat",
        gt_key="indian_pines_gt",
        class_names=(
            "Alfalfa",
            "Corn-notill",
            "Corn-mintill",
            "Corn",
            "Grass-pasture",
            "Grass-trees",
            "Grass-pasture-mowed",
            "Hay-windrowed",
            "Oats",
            "Soybean-notill",
            "Soybean-mintill",
            "Soybean-clean",
            "Wheat",
            "Woods",
            "Buildings-Grass-Trees-Drives",
            "Stone-Steel-Towers",
        ),
    ),
    "SA": DatasetSpec(
        code="SA",
        data_file="Salinas_corrected.mat",
        data_key="salinas_corrected",
        gt_file="Salinas_gt.mat",
        gt_key="salinas_gt",
        class_names=(
            "Brocoli_green_weeds_1",
            "Brocoli_green_weeds_2",
            "Fallow",
            "Fallow_rough_plow",
            "Fallow_smooth",
            "Stubble",
            "Celery",
            "Grapes_untrained",
            "Soil_vinyard_develop",
            "Corn_senesced_green",
            "Lettuce_romaine_4wk",
            "Lettuce_romaine_5wk",
            "Lettuce_romaine_6wk",
            "Lettuce_romaine_7wk",
            "Vinyard_untrained",
            "Vinyard_vertical",
        ),
    ),
    "PU": DatasetSpec(
        code="PU",
        data_file="PaviaU.mat",
        data_key="paviaU",
        gt_file="PaviaU_gt.mat",
        gt_key="paviaU_gt",
        class_names=(
            "Asphalt",
            "Meadows",
            "Gravel",
            "Trees",
            "Painted metal sheets",
            "Bare Soil",
            "Bitumen",
            "Self-Blocking Bricks",
            "Shadows",
        ),
    ),
}


def load_hsi_dataset(name: str, dataset_dir: str | Path = "dataset"):
    """Load a hyperspectral cube and ground-truth labels by dataset code."""
    from scipy.io import loadmat

    code = name.upper()
    if code not in DATASET_SPECS:
        supported = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(f"Unsupported dataset {name!r}. Supported values: {supported}")

    spec = DATASET_SPECS[code]
    dataset_path = Path(dataset_dir)
    data_path = dataset_path / spec.data_file
    gt_path = dataset_path / spec.gt_file

    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing ground-truth file: {gt_path}")

    cube = loadmat(data_path)[spec.data_key]
    gt = loadmat(gt_path)[spec.gt_key]
    return np.asarray(cube, dtype=np.float32), np.asarray(gt, dtype=np.int64), list(spec.class_names)


def apply_pca_cube(cube: np.ndarray, n_components: int):
    """Apply PCA to the spectral dimension of a hyperspectral cube."""
    from sklearn.decomposition import PCA

    flat_cube = np.reshape(cube, (-1, cube.shape[2]))
    pca = PCA(n_components=n_components, whiten=True)
    reduced = pca.fit_transform(flat_cube)
    reduced = np.reshape(reduced, (cube.shape[0], cube.shape[1], n_components))
    return np.asarray(reduced, dtype=np.float32), pca


def sample_ground_truth(gt: np.ndarray, train_rate: float, random_state: int = 100):
    """Split non-background labels into two maps while preserving class ratios."""
    from sklearn.model_selection import train_test_split

    indices = np.nonzero(gt)
    locations = list(zip(*indices))
    labels = gt[indices].ravel()

    if train_rate > 1:
        train_rate = int(train_rate)

    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    train_indices, test_indices = train_test_split(
        locations,
        train_size=train_rate,
        stratify=labels,
        random_state=random_state,
    )
    train_rows, train_cols = zip(*train_indices)
    test_rows, test_cols = zip(*test_indices)
    train_gt[train_rows, train_cols] = gt[train_rows, train_cols]
    test_gt[test_rows, test_cols] = gt[test_rows, test_cols]
    return train_gt, test_gt


def split_ground_truth(
    gt: np.ndarray,
    train_rate: float,
    val_rate: float,
    random_state: int = 100,
):
    """Reproduce the notebook split: train first, then split the remainder into val/test."""
    if not 0 < train_rate < 1:
        raise ValueError("train_rate must be between 0 and 1.")
    if not 0 <= val_rate < 1:
        raise ValueError("val_rate must be between 0 and 1.")
    if train_rate + val_rate >= 1:
        raise ValueError("train_rate + val_rate must be smaller than 1.")

    train_gt, test_gt = sample_ground_truth(gt, train_rate, random_state=random_state)
    if val_rate == 0:
        val_gt = np.zeros_like(gt)
        return train_gt, val_gt, test_gt

    val_ratio_within_remainder = val_rate / (1 - train_rate)
    try:
        val_gt, test_gt = sample_ground_truth(
            test_gt,
            val_ratio_within_remainder,
            random_state=random_state,
        )
    except ValueError as exc:
        raise ValueError(
            "Unable to create a stratified validation split. "
            "Try increasing val_rate or train_rate for rare classes."
        ) from exc
    return train_gt, val_gt, test_gt


def build_sample_report(
    gt: np.ndarray,
    train_gt: np.ndarray,
    val_gt: np.ndarray,
    test_gt: np.ndarray,
    class_names: list[str] | None = None,
) -> str:
    """Format per-class split statistics for logs and docs."""
    header = f"{'class':<4}{'name':<32}{'train':>10}{'val':>10}{'test':>10}{'total':>10}"
    lines = [header]
    for class_id in np.unique(gt):
        if class_id == 0:
            continue
        class_name = class_names[class_id - 1] if class_names else f"class_{class_id}"
        lines.append(
            f"{class_id:<4}{class_name:<32}"
            f"{int((train_gt == class_id).sum()):>10}"
            f"{int((val_gt == class_id).sum()):>10}"
            f"{int((test_gt == class_id).sum()):>10}"
            f"{int((gt == class_id).sum()):>10}"
        )
    lines.append(
        f"{'all':<36}"
        f"{int(np.count_nonzero(train_gt)):>10}"
        f"{int(np.count_nonzero(val_gt)):>10}"
        f"{int(np.count_nonzero(test_gt)):>10}"
        f"{int(np.count_nonzero(gt)):>10}"
    )
    return "\n".join(lines)


class PatchDataset(Dataset):
    """Create spatial-spectral patches from a hyperspectral cube."""

    def __init__(self, cube: np.ndarray, gt: np.ndarray, patch_size: int, is_pred: bool = False):
        super().__init__()
        self.is_pred = is_pred
        self.patch_size = patch_size
        padding = patch_size // 2
        self.data = np.pad(cube, ((padding, padding), (padding, padding), (0, 0)), mode="constant")

        if is_pred:
            gt = np.ones_like(gt)
        self.label = np.pad(gt, (padding, padding), mode="constant")

        rows, cols = np.nonzero(gt)
        rows = rows + padding
        cols = cols + padding
        self.indices = np.asarray(list(zip(rows, cols)), dtype=np.int64)
        if not is_pred:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        row, col = self.indices[index]
        radius = self.patch_size // 2
        row_start, col_start = row - radius, col - radius
        row_end, col_end = row_start + self.patch_size, col_start + self.patch_size

        patch = self.data[row_start:row_end, col_start:col_end]
        patch = np.asarray(patch, dtype=np.float32).transpose((2, 0, 1))
        patch_tensor = torch.from_numpy(patch)

        if self.is_pred:
            return patch_tensor

        label = int(self.label[row, col]) - 1
        return patch_tensor, torch.tensor(label, dtype=torch.long)


def create_dataloaders(
    cube: np.ndarray,
    gt: np.ndarray,
    patch_size: int,
    batch_size: int,
    train_rate: float,
    val_rate: float,
    random_state: int = 100,
    num_workers: int = 0,
):
    """Build train/val/full-image dataloaders from a cube and labels."""
    train_gt, val_gt, test_gt = split_ground_truth(
        gt,
        train_rate=train_rate,
        val_rate=val_rate,
        random_state=random_state,
    )

    train_dataset = PatchDataset(cube, train_gt, patch_size=patch_size)
    val_dataset = PatchDataset(cube, val_gt, patch_size=patch_size)
    pred_dataset = PatchDataset(cube, gt, patch_size=patch_size, is_pred=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    pred_loader = DataLoader(
        pred_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return {
        "train_gt": train_gt,
        "val_gt": val_gt,
        "test_gt": test_gt,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "pred_loader": pred_loader,
    }
