"""Training loop utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from .utils import ensure_dir, save_json


@dataclass
class TrainingConfig:
    epochs: int
    lr: float
    weight_decay: float
    eval_interval: int
    output_dir: Path
    keep_all_checkpoints: bool = False


@dataclass
class FitResult:
    best_checkpoint: Path
    best_val_acc: float
    history: dict


def train_epoch(
    model: torch.nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_examples += batch_size
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()

    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def evaluate_accuracy(model: torch.nn.Module, loader, device: torch.device) -> float:
    model.eval()
    total_correct = 0
    total_examples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        total_examples += targets.size(0)
        total_correct += (logits.argmax(dim=1) == targets).sum().item()

    if total_examples == 0:
        return 0.0
    return total_correct / total_examples


def save_training_curves(history: dict, output_path: str | Path) -> Path:
    """Save a compact training summary figure."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))
    ax_loss.plot(epochs, history["train_loss"], label="train_loss")
    ax_loss.set_title("Training Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")

    ax_acc.plot(epochs, history["train_acc"], label="train_acc")
    if history["val_epoch"]:
        ax_acc.plot(history["val_epoch"], history["val_acc"], label="val_acc")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def fit(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    config: TrainingConfig,
) -> FitResult:
    """Train a model and persist the best validation checkpoint."""
    output_dir = ensure_dir(config.output_dir)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_acc": [], "val_epoch": []}
    best_checkpoint: Path | None = None
    best_val_acc = float("-inf")
    checkpoints: list[Path] = []

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        print(f"epoch {epoch}/{config.epochs} loss={train_loss:.6f} train_acc={train_acc:.4f}")

        should_eval = (epoch % config.eval_interval == 0) or (epoch == config.epochs)
        if not should_eval:
            continue

        val_acc = evaluate_accuracy(model, val_loader, device)
        history["val_acc"].append(val_acc)
        history["val_epoch"].append(epoch)
        checkpoint = output_dir / f"epoch_{epoch:03d}_valacc_{val_acc:.4f}.pth"
        torch.save(model.state_dict(), checkpoint)
        checkpoints.append(checkpoint)
        print(f"epoch {epoch}/{config.epochs} val_acc={val_acc:.4f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_checkpoint = checkpoint

    if best_checkpoint is None:
        raise RuntimeError("Training finished without producing a checkpoint.")

    if not config.keep_all_checkpoints:
        for checkpoint in checkpoints:
            if checkpoint != best_checkpoint and checkpoint.exists():
                checkpoint.unlink()

    save_json(history, output_dir / "training_history.json")
    save_training_curves(history, output_dir / "training_curves.png")
    return FitResult(best_checkpoint=best_checkpoint, best_val_acc=best_val_acc, history=history)
