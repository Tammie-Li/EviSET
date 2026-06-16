"""EEG-Inception baseline method.

The network uses parallel temporal filters with different receptive fields,
spatial filtering over electrodes, and separable inception blocks for ERP-like
EEG decoding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from task.train import BaseMethod, ExperimentContext


class InceptionSpatialBranch(nn.Module):
    def __init__(self, eeg_channels: int, out_channels: int, temporal_kernel: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                1,
                out_channels,
                kernel_size=(1, temporal_kernel),
                padding=(0, temporal_kernel // 2),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(eeg_channels, 1),
                groups=out_channels,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 2)),
            nn.Dropout(0.35),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SeparableTemporalBranch(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=(1, kernel_size),
                padding=(0, kernel_size // 2),
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EEGInception(nn.Module):
    def __init__(self, eeg_channels: int, eeg_samples: int, num_classes: int = 3) -> None:
        super().__init__()
        branch_channels = 8
        self.first_inception = nn.ModuleList(
            [
                InceptionSpatialBranch(eeg_channels, branch_channels, temporal_kernel=31),
                InceptionSpatialBranch(eeg_channels, branch_channels, temporal_kernel=15),
                InceptionSpatialBranch(eeg_channels, branch_channels, temporal_kernel=7),
            ]
        )
        in_channels = branch_channels * len(self.first_inception)
        self.second_inception = nn.ModuleList(
            [
                SeparableTemporalBranch(in_channels, 16, kernel_size=15),
                SeparableTemporalBranch(in_channels, 16, kernel_size=7),
                SeparableTemporalBranch(in_channels, 16, kernel_size=3),
            ]
        )
        self.post = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 2)),
            nn.Dropout(0.4),
            nn.Conv2d(48, 32, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, eeg_channels, eeg_samples)
            feature_dim = self.forward_features(dummy).shape[1]
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward_features(self, eeg: torch.Tensor) -> torch.Tensor:
        x = eeg.unsqueeze(1)
        x = torch.cat([branch(x) for branch in self.first_inception], dim=1)
        x = torch.cat([branch(x) for branch in self.second_inception], dim=1)
        x = self.post(x)
        return torch.flatten(x, start_dim=1)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(eeg))


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float


class EEGInceptionMethod(BaseMethod):
    method_type = "EEG-only"
    name = "EEGInception"
    year = "2020"

    def config(self) -> TrainingConfig:
        if self.quick:
            return TrainingConfig(epochs=7, batch_size=256, learning_rate=1e-3, weight_decay=1e-4)
        return TrainingConfig(epochs=32, batch_size=256, learning_rate=8e-4, weight_decay=1e-4)

    def predict(self, ctx: ExperimentContext) -> np.ndarray:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        eeg = ctx.features.raw_eeg.astype(np.float32)
        train_eeg = eeg[ctx.train_idx]
        test_eeg = eeg[ctx.test_idx]

        cfg = self.config()
        model = EEGInception(train_eeg.shape[1], train_eeg.shape[2], num_classes=3).to(device)
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
                    loss = criterion(model(eeg_batch), y_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    progress.update(epoch, loss.detach())

        model.eval()
        preds = []
        test_loader = DataLoader(TensorDataset(torch.from_numpy(test_eeg)), batch_size=cfg.batch_size)
        with torch.no_grad():
            for (eeg_batch,) in test_loader:
                preds.append(model(eeg_batch.to(device)).argmax(dim=1).cpu().numpy())
        return np.concatenate(preds, axis=0)
