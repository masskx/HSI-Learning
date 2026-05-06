"""HybridSN implementation extracted from the original notebook."""

from __future__ import annotations

import torch
import torch.nn as nn


class HybridSN(nn.Module):
    """3D-2D convolutional baseline for hyperspectral classification."""

    def __init__(self, input_channels: int, patch_size: int, num_classes: int):
        super().__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(7, 3, 3)),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3)),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3)),
            nn.ReLU(inplace=True),
        )

        conv3d_shape = self._shape_after_3dconv()
        flattened_channels = conv3d_shape[1] * conv3d_shape[2]
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=flattened_channels, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
        )

        feature_dim = self._shape_after_2dconv(conv3d_shape)
        self.dense1 = nn.Sequential(nn.Linear(feature_dim, 256), nn.ReLU(inplace=True), nn.Dropout(p=0.4))
        self.dense2 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(p=0.4))
        self.dense3 = nn.Linear(128, num_classes)

    def _shape_after_3dconv(self):
        sample = torch.zeros((1, 1, self.input_channels, self.patch_size, self.patch_size))
        with torch.no_grad():
            sample = self.conv1(sample)
            sample = self.conv2(sample)
            sample = self.conv3(sample)
        return sample.shape

    def _shape_after_2dconv(self, conv3d_shape) -> int:
        sample = torch.zeros((1, conv3d_shape[1] * conv3d_shape[2], conv3d_shape[3], conv3d_shape[4]))
        with torch.no_grad():
            sample = self.conv4(sample)
        return int(sample.shape[1] * sample.shape[2] * sample.shape[3])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = inputs.unsqueeze(1)
        features = self.conv1(features)
        features = self.conv2(features)
        features = self.conv3(features)
        features = features.view(features.shape[0], features.shape[1] * features.shape[2], features.shape[3], features.shape[4])
        features = self.conv4(features)
        features = features.contiguous().view(features.shape[0], -1)
        features = self.dense1(features)
        features = self.dense2(features)
        return self.dense3(features)
