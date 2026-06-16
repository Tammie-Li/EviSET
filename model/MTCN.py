"""MTCN EEG-only baseline method.

The implementation keeps the method-specific logic in this file. It follows
the MTCN idea of combining the supervised EEG classification task with two
task-related self-supervised tasks: masked temporal recognition (MTR) and
masked spatial recognition (MSR). The network explicitly separates
task-specific and task-shared features and applies a structured decorrelation
loss between them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from task.train import BaseMethod, ExperimentContext


def _region_indices(length: int, regions: int) -> List[Tuple[int, int]]:
    bounds = np.linspace(0, length, regions + 1).round().astype(int)
    return [(int(bounds[i]), int(bounds[i + 1])) for i in range(regions)]


def _channel_region_indices(channels: int, regions: int) -> List[Tuple[int, int]]:
    if channels == 64 and regions == 8:
        return [
            (0, 7),    # Fpz/Fp/AF
            (7, 16),   # F
            (16, 25),  # FC/FT
            (25, 34),  # C/T
            (34, 42),  # CP/TP
            (42, 49),  # P
            (49, 59),  # PO/O
            (59, 64),  # auxiliary channels recorded with this cap montage
        ]
    return _region_indices(channels, regions)


def mask_temporal_regions(eeg: torch.Tensor, labels: torch.Tensor, regions: int = 9) -> torch.Tensor:
    """Mask one temporal region per sample with Gaussian noise for MTR."""

    masked = eeg.clone()
    sample_std = eeg.std(dim=(1, 2), keepdim=True).clamp_min(1e-4)
    for i, region in enumerate(labels.tolist()):
        start, end = _region_indices(eeg.shape[2], regions)[region]
        masked[i, :, start:end] = torch.randn_like(masked[i, :, start:end]) * sample_std[i]
    return masked


def mask_spatial_regions(eeg: torch.Tensor, labels: torch.Tensor, regions: int = 8) -> torch.Tensor:
    """Mask one channel group per sample with Gaussian noise for MSR."""

    masked = eeg.clone()
    sample_std = eeg.std(dim=(1, 2), keepdim=True).clamp_min(1e-4)
    channel_regions = _channel_region_indices(eeg.shape[1], regions)
    for i, region in enumerate(labels.tolist()):
        start, end = channel_regions[region]
        masked[i, start:end, :] = torch.randn_like(masked[i, start:end, :]) * sample_std[i]
    return masked


def structured_decorrelation_loss(specific: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
    specific_flat = F.normalize(torch.flatten(specific, start_dim=1), dim=1)
    shared_flat = F.normalize(torch.flatten(shared, start_dim=1), dim=1)
    return torch.mean(torch.sum(specific_flat * shared_flat, dim=1).pow(2))


class TaskSpecificExtractor(nn.Module):
    """Temporal filtering followed by depthwise spatial filtering."""

    def __init__(
        self,
        eeg_channels: int,
        f1: int = 8,
        depth_multiplier: int = 2,
        temporal_kernel: int = 31,
        dropout: float = 0.45,
    ) -> None:
        super().__init__()
        f2 = f1 * depth_multiplier
        self.net = nn.Sequential(
            nn.Conv2d(
                1,
                f1,
                kernel_size=(1, temporal_kernel),
                padding=(0, temporal_kernel // 2),
                bias=False,
            ),
            nn.BatchNorm2d(f1),
            nn.Conv2d(
                f1,
                f2,
                kernel_size=(eeg_channels, 1),
                groups=f1,
                bias=False,
            ),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.net(eeg.unsqueeze(1))


class TaskSharedExtractor(nn.Module):
    """Shared separable temporal extractor used by all MTCN tasks."""

    def __init__(self, feature_channels: int, separable_kernel: int = 15, dropout: float = 0.35) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                feature_channels,
                feature_channels,
                kernel_size=(1, separable_kernel),
                padding=(0, separable_kernel // 2),
                groups=feature_channels,
                bias=False,
            ),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.ELU(),
            nn.Dropout(dropout),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class MTCNNetwork(nn.Module):
    """Multi-task collaborative network for EEG classification."""

    def __init__(
        self,
        eeg_channels: int,
        eeg_samples: int,
        num_classes: int = 3,
        temporal_regions: int = 9,
        spatial_regions: int = 8,
        f1: int = 8,
        depth_multiplier: int = 2,
    ) -> None:
        super().__init__()
        feature_channels = f1 * depth_multiplier
        self.primary_specific = TaskSpecificExtractor(eeg_channels, f1, depth_multiplier)
        self.temporal_specific = TaskSpecificExtractor(eeg_channels, f1, depth_multiplier)
        self.spatial_specific = TaskSpecificExtractor(eeg_channels, f1, depth_multiplier)
        self.shared_extractor = TaskSharedExtractor(feature_channels)

        with torch.no_grad():
            dummy = torch.zeros(1, eeg_channels, eeg_samples)
            feature_dim = self._task_representation(dummy, self.primary_specific)[0].shape[1]

        self.primary_head = nn.Linear(feature_dim, num_classes)
        self.temporal_head = nn.Linear(feature_dim, temporal_regions)
        self.spatial_head = nn.Linear(feature_dim, spatial_regions)
        self.log_vars = nn.Parameter(torch.zeros(6))

    def _task_representation(
        self, eeg: torch.Tensor, extractor: TaskSpecificExtractor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        specific = extractor(eeg)
        shared = self.shared_extractor(specific)
        representation = torch.flatten(torch.cat([specific, shared], dim=1), start_dim=1)
        return representation, specific, shared

    def primary_logits(self, eeg: torch.Tensor) -> torch.Tensor:
        representation, _, _ = self._task_representation(eeg, self.primary_specific)
        return self.primary_head(representation)

    def forward(
        self,
        eeg: torch.Tensor,
        temporal_masked_eeg: torch.Tensor,
        spatial_masked_eeg: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        primary_repr, primary_specific, primary_shared = self._task_representation(eeg, self.primary_specific)
        temporal_repr, temporal_specific, temporal_shared = self._task_representation(
            temporal_masked_eeg, self.temporal_specific
        )
        spatial_repr, spatial_specific, spatial_shared = self._task_representation(
            spatial_masked_eeg, self.spatial_specific
        )
        return (
            self.primary_head(primary_repr),
            self.temporal_head(temporal_repr),
            self.spatial_head(spatial_repr),
            (primary_specific, primary_shared),
            (temporal_specific, temporal_shared),
            (spatial_specific, spatial_shared),
        )

    def uncertainty_weighted_loss(self, losses: List[torch.Tensor]) -> torch.Tensor:
        total = torch.zeros((), device=losses[0].device)
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * loss + self.log_vars[i]
        return total


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    temporal_regions: int = 9
    spatial_regions: int = 8


class MTCNMethod(BaseMethod):
    method_type = "EEG-only"
    name = "MTCN"
    year = "2024"

    def config(self) -> TrainingConfig:
        if self.quick:
            return TrainingConfig(epochs=6, batch_size=256, learning_rate=1e-3, weight_decay=1e-4)
        return TrainingConfig(epochs=28, batch_size=256, learning_rate=8e-4, weight_decay=1e-4)

    def predict(self, ctx: ExperimentContext) -> np.ndarray:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        eeg = ctx.features.raw_eeg.astype(np.float32)
        train_eeg = eeg[ctx.train_idx]
        test_eeg = eeg[ctx.test_idx]

        cfg = self.config()
        model = MTCNNetwork(
            eeg_channels=train_eeg.shape[1],
            eeg_samples=train_eeg.shape[2],
            num_classes=3,
            temporal_regions=cfg.temporal_regions,
            spatial_regions=cfg.spatial_regions,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        criterion = nn.CrossEntropyLoss()

        train_ds = TensorDataset(
            torch.from_numpy(train_eeg),
            torch.from_numpy(ctx.y_train.astype(np.int64)),
        )
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, generator=generator)

        model.train()
        with self.training_progress(cfg.epochs, len(train_loader)) as progress:
            for epoch in range(1, cfg.epochs + 1):
                for eeg_batch, y_batch in train_loader:
                    eeg_batch = eeg_batch.to(device)
                    y_batch = y_batch.to(device)
                    batch_size = eeg_batch.shape[0]

                    temporal_labels = torch.randint(
                        low=0,
                        high=cfg.temporal_regions,
                        size=(batch_size,),
                        device=device,
                    )
                    spatial_labels = torch.randint(
                        low=0,
                        high=cfg.spatial_regions,
                        size=(batch_size,),
                        device=device,
                    )
                    temporal_masked = mask_temporal_regions(eeg_batch, temporal_labels, cfg.temporal_regions)
                    spatial_masked = mask_spatial_regions(eeg_batch, spatial_labels, cfg.spatial_regions)

                    (
                        primary_logits,
                        temporal_logits,
                        spatial_logits,
                        primary_features,
                        temporal_features,
                        spatial_features,
                    ) = model(eeg_batch, temporal_masked, spatial_masked)

                    losses = [
                        criterion(primary_logits, y_batch),
                        criterion(temporal_logits, temporal_labels),
                        criterion(spatial_logits, spatial_labels),
                        structured_decorrelation_loss(*primary_features),
                        structured_decorrelation_loss(*temporal_features),
                        structured_decorrelation_loss(*spatial_features),
                    ]
                    loss = model.uncertainty_weighted_loss(losses)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    progress.update(epoch, loss.detach())

        model.eval()
        preds = []
        test_ds = TensorDataset(torch.from_numpy(test_eeg))
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
        with torch.no_grad():
            for (eeg_batch,) in test_loader:
                logits = model.primary_logits(eeg_batch.to(device))
                preds.append(logits.argmax(dim=1).cpu().numpy())
        return np.concatenate(preds, axis=0)
