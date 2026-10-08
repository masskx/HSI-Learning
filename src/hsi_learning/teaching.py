"""Small, testable primitives shared by lecture notebooks and CLI demos.

Protocol decisions are explicit. No function here chooses a threshold or a
checkpoint using test labels. Positions are flattened row-major pixel indices.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PatchEncoder(nn.Module):
    """The ch06 convolutional backbone, without a classifier."""
    def __init__(self, channels=12):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        return self.features(x).flatten(1)


class PatchClassifier(nn.Module):
    def __init__(self, channels, classes):
        super().__init__()
        self.encoder = PatchEncoder(channels)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(.3), nn.Linear(64, classes))

    def forward(self, x):
        return self.head(self.encoder(x))


def fit_preprocessing(cube, fit_ids, components=12):
    """Fit only on declared training centers; return portable numpy parameters."""
    flat = cube.reshape(-1, cube.shape[-1]).astype(np.float32)
    ids = np.asarray(fit_ids, dtype=np.int64)
    if not len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError('fit_ids must be nonempty and unique')
    pca = PCA(n_components=components, whiten=True, svd_solver='full').fit(flat[ids])
    z_train = pca.transform(flat[ids])
    scaler = StandardScaler().fit(z_train)
    params = dict(pca_mean=pca.mean_, components=pca.components_,
                  variance=pca.explained_variance_, scale_mean=scaler.mean_, scale=scaler.scale_)
    return transform_cube(cube, params), params


def transform_cube(cube, params):
    flat = cube.reshape(-1, cube.shape[-1]).astype(np.float32)
    z = (flat - params['pca_mean']) @ params['components'].T
    z /= np.sqrt(np.maximum(params['variance'], np.finfo(np.float32).eps))
    z = (z - params['scale_mean']) / params['scale']
    return z.astype(np.float32).reshape(*cube.shape[:2], -1)


def patches_at(cube, ids, size=9):
    if size < 3 or size % 2 != 1:
        raise ValueError('patch size must be odd and >=3')
    radius = size // 2
    padded = np.pad(cube, ((radius, radius), (radius, radius), (0, 0)))
    positions = np.column_stack(np.unravel_index(np.asarray(ids, dtype=int), cube.shape[:2]))
    return torch.from_numpy(np.stack([padded[r:r+size, c:c+size].transpose(2, 0, 1)
                                     for r, c in positions]).astype(np.float32))


@dataclass
class Episode:
    classes: np.ndarray
    support_ids: np.ndarray
    query_ids: np.ndarray
    support_y: np.ndarray
    query_y: np.ndarray


class EpisodeSampler:
    def __init__(self, labels, pool_ids, classes, seed, capacity):
        self.rng = np.random.default_rng(seed)
        self.classes = np.asarray(classes, dtype=int)
        self.pools = {int(c): np.asarray(pool_ids)[labels[np.asarray(pool_ids)] == c]
                      for c in self.classes}
        insufficient = {c: len(v) for c, v in self.pools.items() if len(v) < capacity}
        if insufficient:
            raise ValueError(f'Need {capacity} distinct pixels per class; insufficient: {insufficient}')

    def sample(self, way, shot, query):
        if way < 2 or way > len(self.classes) or min(shot, query) < 1:
            raise ValueError('Invalid episode dimensions; never silently reduce N-way')
        selected = self.rng.choice(self.classes, way, replace=False)
        support, queries = [], []
        for c in selected:
            pool = self.pools[int(c)]
            if len(pool) < shot + query:
                raise ValueError(f'class {c} cannot supply disjoint support/query')
            ids = self.rng.choice(pool, shot + query, replace=False)
            support.extend(ids[:shot]); queries.extend(ids[shot:])
        return Episode(selected, np.asarray(support), np.asarray(queries),
                       np.repeat(np.arange(way), shot), np.repeat(np.arange(way), query))


def prototypes(embeddings, labels, classes):
    """Labels must travel with embeddings, including when the loader shuffles."""
    means = []
    for c in classes:
        values = embeddings[labels == c]
        if len(values) == 0:
            raise ValueError(f'No support for class {c}')
        means.append(values.mean(0))
    return torch.stack(means)


def prototype_logits(support, labels, query, way):
    centers = prototypes(support, labels, range(way))
    return -((query[:, None] - centers[None]) ** 2).sum(-1)


def calibrate_threshold(known_validation_scores, acceptance=.95):
    """Higher score means more unknown. Accept score <= quantile of known val."""
    scores = np.asarray(known_validation_scores)
    if not len(scores) or not np.isfinite(scores).all() or not 0 < acceptance < 1:
        raise ValueError('Finite known validation scores and acceptance in (0,1) required')
    return float(np.quantile(scores, acceptance, method='higher'))


def unknown_scores(logits, embeddings, centers):
    msp = 1 - logits.softmax(-1).max(-1).values
    distance = torch.cdist(embeddings, centers).min(-1).values
    return msp, distance


def rejection_predictions(known_prediction_ids, scores, threshold, unknown_id):
    pred = np.asarray(known_prediction_ids).copy()
    pred[np.asarray(scores) > threshold] = unknown_id
    return pred


def grad_cam(model, layer, inputs, targets=None):
    """Class-specific Grad-CAM: alpha_k = spatial mean(d logit_c / d A_k).

    Returns normalized maps at input resolution and logits. Does not detach
    activations before differentiating; cleans hooks even when forward fails.
    """
    training = model.training
    activations = []
    handle = layer.register_forward_hook(lambda module, args, output: activations.append(output))
    try:
        model.eval()
        with torch.enable_grad():
            x = inputs.detach().clone().requires_grad_(True)
            logits = model(x)
            selected = logits.argmax(-1) if targets is None else targets.to(logits.device)
            score = logits.gather(1, selected[:, None]).sum()
            feature = activations[-1]
            gradient, = torch.autograd.grad(score, feature)
            weights = gradient.mean(dim=(-2, -1), keepdim=True)
            cam = (weights * feature).sum(1, keepdim=True).relu()
            cam = F.interpolate(cam, size=inputs.shape[-2:], mode='bilinear', align_corners=False)
            maximum = cam.flatten(1).amax(1)[:, None, None, None]
            cam = cam / maximum.clamp_min(1e-12)
        return cam[:, 0].detach(), logits.detach()
    finally:
        handle.remove()
        model.train(training)
