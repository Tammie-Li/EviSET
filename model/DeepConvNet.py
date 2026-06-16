"""DeepConvNet baseline method.

Implements the end-to-end EEG Deep ConvNet introduced by Schirrmeister et al.
The model starts with temporal and spatial convolutions and then applies a
stack of convolution-pooling blocks directly on raw EEG windows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from task.train import BaseMethod, ExperimentContext


class ConvPoolBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, kernel_size),
                padding=(0, kernel_size // 2),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepConvNet(nn.Module):
    def __init__(
        self,
        eeg_channels: int,
        eeg_samples: int,
        num_classes: int = 3,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.first_block = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 9), padding=(0, 4), bias=False),
            nn.Conv2d(25, 25, kernel_size=(eeg_channels, 1), bias=False),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(
            ConvPoolBlock(25, 50, kernel_size=7, dropout=dropout),
            ConvPoolBlock(50, 100, kernel_size=5, dropout=dropout),
            ConvPoolBlock(100, 200, kernel_size=5, dropout=dropout),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, eeg_channels, eeg_samples)
            feature_dim = self.forward_features(dummy).shape[1]
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward_features(self, eeg: torch.Tensor) -> torch.Tensor:
        x = eeg.unsqueeze(1)
        x = self.first_block(x)
        x = self.blocks(x)
        return torch.flatten(x, start_dim=1)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(eeg))


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float


class DeepConvNetMethod(BaseMethod):
    method_type = "EEG-only"
    name = "DeepConvNet"
    year = "2017"

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
        model = DeepConvNet(train_eeg.shape[1], train_eeg.shape[2], num_classes=3).to(device)
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
