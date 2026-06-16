"""TCN eye-only baseline method over At/dt eye-movement features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from task.train import BaseMethod, ExperimentContext


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if y.shape[-1] > x.shape[-1]:
            y = y[:, :, : x.shape[-1]]
        return x + y


class EyeTCN(nn.Module):
    def __init__(self, in_channels: int = 2, num_classes: int = 3) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, 48, kernel_size=1),
            nn.BatchNorm1d(48),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            ResidualTCNBlock(48, dilation=1, dropout=0.2),
            ResidualTCNBlock(48, dilation=2, dropout=0.2),
            ResidualTCNBlock(48, dilation=4, dropout=0.2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(48, 48),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(48, num_classes),
        )

    def forward(self, eye_seq: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(eye_seq)
        x = self.blocks(x)
        return self.head(x)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float


class TCNMethod(BaseMethod):
    method_type = "Eye-only"
    name = "TCN"
    year = "2023"

    def config(self) -> TrainingConfig:
        if self.quick:
            return TrainingConfig(epochs=8, batch_size=256, learning_rate=1e-3, weight_decay=1e-4)
        return TrainingConfig(epochs=30, batch_size=256, learning_rate=8e-4, weight_decay=1e-4)

    def predict(self, ctx: ExperimentContext) -> np.ndarray:
        torch.manual_seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        eye_seq = ctx.features.eye_at_dt_sequence
        train_eye = eye_seq[ctx.train_idx]
        test_eye = eye_seq[ctx.test_idx]

        cfg = self.config()
        model = EyeTCN(in_channels=train_eye.shape[1], num_classes=3).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        criterion = nn.CrossEntropyLoss()

        train_ds = TensorDataset(
            torch.from_numpy(train_eye),
            torch.from_numpy(ctx.y_train.astype(np.int64)),
        )
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, generator=generator)

        model.train()
        with self.training_progress(cfg.epochs, len(train_loader)) as progress:
            for epoch in range(1, cfg.epochs + 1):
                for eye_batch, y_batch in train_loader:
                    eye_batch = eye_batch.to(device)
                    y_batch = y_batch.to(device)
                    loss = criterion(model(eye_batch), y_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    progress.update(epoch, loss.detach())

        model.eval()
        preds = []
        test_loader = DataLoader(TensorDataset(torch.from_numpy(test_eye)), batch_size=cfg.batch_size)
        with torch.no_grad():
            for (eye_batch,) in test_loader:
                preds.append(model(eye_batch.to(device)).argmax(dim=1).cpu().numpy())
        return np.concatenate(preds, axis=0)
