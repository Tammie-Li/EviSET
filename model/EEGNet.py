"""EEGNet baseline method.

This implementation follows the EEGNet design: temporal convolution,
depthwise spatial convolution, separable temporal convolution, and a compact
classification head trained directly on the raw EEG window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from task.train import BaseMethod, ExperimentContext


class EEGNet(nn.Module):
    """Compact EEGNet architecture for input shaped as (batch, channels, time)."""

    def __init__(
        self,
        eeg_channels: int,
        eeg_samples: int,
        num_classes: int = 3,
        f1: int = 8,
        depth_multiplier: int = 2,
        temporal_kernel: int = 31,
        separable_kernel: int = 15,
        dropout: float = 0.5,
        feature_dim: int = 0,
    ) -> None:
        super().__init__()
        f2 = f1 * depth_multiplier
        self.feature_dim = feature_dim

        self.temporal_conv = nn.Sequential(
            nn.Conv2d(
                1,
                f1,
                kernel_size=(1, temporal_kernel),
                padding=(0, temporal_kernel // 2),
                bias=False,
            ),
            nn.BatchNorm2d(f1),
        )
        self.depthwise_spatial_conv = nn.Sequential(
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
        self.separable_conv = nn.Sequential(
            nn.Conv2d(
                f2,
                f2,
                kernel_size=(1, separable_kernel),
                padding=(0, separable_kernel // 2),
                groups=f2,
                bias=False,
            ),
            nn.Conv2d(f2, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, eeg_channels, eeg_samples)
            conv_dim = self.forward_features(dummy).shape[1]
        if feature_dim > 0:
            self.feature_encoder = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ELU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ELU(),
            )
            classifier_dim = conv_dim + 64
        else:
            self.feature_encoder = None
            classifier_dim = conv_dim
        self.classifier = nn.Linear(classifier_dim, num_classes)

    def forward_features(self, eeg: torch.Tensor) -> torch.Tensor:
        x = eeg.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.depthwise_spatial_conv(x)
        x = self.separable_conv(x)
        return torch.flatten(x, start_dim=1)

    def forward(self, eeg: torch.Tensor, features: torch.Tensor | None = None) -> torch.Tensor:
        x = self.forward_features(eeg)
        if self.feature_encoder is not None:
            if features is None:
                raise ValueError("EEG descriptor features are required for this EEGNet instance.")
            x = torch.cat([x, self.feature_encoder(features)], dim=1)
        return self.classifier(x)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float


class EEGNetMethod(BaseMethod):
    method_type = "EEG-only"
    name = "EEGNet"
    year = "2018"

    def config(self) -> TrainingConfig:
        if self.quick:
            return TrainingConfig(epochs=8, batch_size=256, learning_rate=1e-3, weight_decay=1e-4)
        return TrainingConfig(epochs=35, batch_size=256, learning_rate=8e-4, weight_decay=1e-4)

    def predict(self, ctx: ExperimentContext) -> np.ndarray:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        eeg = ctx.features.raw_eeg.astype(np.float32)
        train_eeg = eeg[ctx.train_idx]
        test_eeg = eeg[ctx.test_idx]
        feature_scaler = StandardScaler()
        train_features = feature_scaler.fit_transform(ctx.features.eeg[ctx.train_idx]).astype(np.float32)
        test_features = feature_scaler.transform(ctx.features.eeg[ctx.test_idx]).astype(np.float32)

        model = EEGNet(
            eeg_channels=train_eeg.shape[1],
            eeg_samples=train_eeg.shape[2],
            num_classes=3,
            feature_dim=train_features.shape[1],
        ).to(device)
        cfg = self.config()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        criterion = nn.CrossEntropyLoss()

        train_ds = TensorDataset(
            torch.from_numpy(train_eeg),
            torch.from_numpy(train_features),
            torch.from_numpy(ctx.y_train.astype(np.int64)),
        )
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, generator=generator)

        model.train()
        with self.training_progress(cfg.epochs, len(train_loader)) as progress:
            for epoch in range(1, cfg.epochs + 1):
                for eeg_batch, feature_batch, y_batch in train_loader:
                    eeg_batch = eeg_batch.to(device)
                    feature_batch = feature_batch.to(device)
                    y_batch = y_batch.to(device)

                    logits = model(eeg_batch, feature_batch)
                    loss = criterion(logits, y_batch)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    progress.update(epoch, loss.detach())

        model.eval()
        preds = []
        test_ds = TensorDataset(torch.from_numpy(test_eeg), torch.from_numpy(test_features))
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
        with torch.no_grad():
            for eeg_batch, feature_batch in test_loader:
                logits = model(eeg_batch.to(device), feature_batch.to(device))
                preds.append(logits.argmax(dim=1).cpu().numpy())
        return np.concatenate(preds, axis=0)
