"""MSNet multimodal fusion baseline method.

Implements a multimodal self-attention network for EEG and eye-tracking data.
Each modality is encoded separately and the modality tokens are fused by
self-attention before classification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from task.train import BaseMethod, ExperimentContext


def build_eye_sequence(eye: np.ndarray) -> np.ndarray:
    dist = np.sqrt((eye[:, :, 0] - eye[:, :, 2]) ** 2 + (eye[:, :, 1] - eye[:, :, 3]) ** 2)
    a_t = (dist < 70.0).astype(np.float32)
    d_t = np.clip(dist / 70.0, 0.0, 5.0).astype(np.float32)
    return np.stack([a_t, d_t], axis=1).astype(np.float32)


class EyeEncoder(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Conv1d(32, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, eye: torch.Tensor) -> torch.Tensor:
        return self.net(eye).squeeze(-1)


class EEGEncoder(nn.Module):
    def __init__(self, eeg_channels: int, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 15), padding=(0, 7), bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=(eeg_channels, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Conv2d(16, d_model, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(d_model),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.net(eeg.unsqueeze(1)).flatten(start_dim=1)


class MSNet(nn.Module):
    def __init__(self, eeg_channels: int, num_classes: int = 3, d_model: int = 48) -> None:
        super().__init__()
        self.eye_encoder = EyeEncoder(d_model=d_model)
        self.eeg_encoder = EEGEncoder(eeg_channels=eeg_channels, d_model=d_model)
        self.modality_pos = nn.Parameter(torch.zeros(1, 2, d_model))
        self.attention = nn.MultiheadAttention(d_model, num_heads=4, dropout=0.2, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, eeg: torch.Tensor, eye: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack([self.eeg_encoder(eeg), self.eye_encoder(eye)], dim=1)
        tokens = tokens + self.modality_pos
        tokens, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        fused = self.norm(tokens.mean(dim=1))
        return self.head(fused)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float


class MSNetMethod(BaseMethod):
    method_type = "Fusion"
    name = "MSNet"
    year = "2025"

    def config(self) -> TrainingConfig:
        if self.quick:
            return TrainingConfig(epochs=7, batch_size=256, learning_rate=1e-3, weight_decay=1e-4)
        return TrainingConfig(epochs=30, batch_size=256, learning_rate=8e-4, weight_decay=1e-4)

    def predict(self, ctx: ExperimentContext) -> np.ndarray:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        eeg = ctx.features.raw_eeg.astype(np.float32)
        eye = build_eye_sequence(ctx.features.raw_eye)
        train_eeg, test_eeg = eeg[ctx.train_idx], eeg[ctx.test_idx]
        train_eye, test_eye = eye[ctx.train_idx], eye[ctx.test_idx]

        cfg = self.config()
        model = MSNet(eeg_channels=train_eeg.shape[1], num_classes=3).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        criterion = nn.CrossEntropyLoss()

        train_ds = TensorDataset(
            torch.from_numpy(train_eeg),
            torch.from_numpy(train_eye),
            torch.from_numpy(ctx.y_train.astype(np.int64)),
        )
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, generator=generator)

        model.train()
        with self.training_progress(cfg.epochs, len(train_loader)) as progress:
            for epoch in range(1, cfg.epochs + 1):
                for eeg_batch, eye_batch, y_batch in train_loader:
                    eeg_batch = eeg_batch.to(device)
                    eye_batch = eye_batch.to(device)
                    y_batch = y_batch.to(device)
                    loss = criterion(model(eeg_batch, eye_batch), y_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    progress.update(epoch, loss.detach())

        model.eval()
        preds = []
        test_ds = TensorDataset(torch.from_numpy(test_eeg), torch.from_numpy(test_eye))
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)
        with torch.no_grad():
            for eeg_batch, eye_batch in test_loader:
                logits = model(eeg_batch.to(device), eye_batch.to(device))
                preds.append(logits.argmax(dim=1).cpu().numpy())
        return np.concatenate(preds, axis=0)
