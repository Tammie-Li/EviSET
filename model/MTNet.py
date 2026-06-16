"""MTNet multimodal fusion baseline method.

Implements an intermediate-fusion multimodal Transformer network for EEG and
eye tracking. Local encoders form modality tokens and a Transformer encoder
models global cross-modal dependencies.
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


class EEGTokenEncoder(nn.Module):
    def __init__(self, d_model: int = 48) -> None:
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.Conv1d(16, d_model, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        batch, channels, samples = eeg.shape
        x = eeg.reshape(batch * channels, 1, samples)
        x = self.local(x).squeeze(-1)
        return x.reshape(batch, channels, -1)


class EyeTokenEncoder(nn.Module):
    def __init__(self, d_model: int = 48, tokens: int = 4) -> None:
        super().__init__()
        self.tokens = tokens
        self.local = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Conv1d(32, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ELU(),
        )

    def forward(self, eye: torch.Tensor) -> torch.Tensor:
        x = self.local(eye)
        x = nn.functional.adaptive_avg_pool1d(x, self.tokens)
        return x.transpose(1, 2)


class MTNet(nn.Module):
    def __init__(
        self,
        eeg_channels: int,
        num_classes: int = 3,
        d_model: int = 48,
        eye_tokens: int = 4,
    ) -> None:
        super().__init__()
        self.eeg_encoder = EEGTokenEncoder(d_model=d_model)
        self.eye_encoder = EyeTokenEncoder(d_model=d_model, tokens=eye_tokens)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, 1 + eeg_channels + eye_tokens, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=128,
            dropout=0.25,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, eeg: torch.Tensor, eye: torch.Tensor) -> torch.Tensor:
        eeg_tokens = self.eeg_encoder(eeg)
        eye_tokens = self.eye_encoder(eye)
        cls = self.cls_token.expand(eeg.shape[0], -1, -1)
        tokens = torch.cat([cls, eeg_tokens, eye_tokens], dim=1)
        tokens = tokens + self.pos[:, : tokens.shape[1], :]
        tokens = self.transformer(tokens)
        return self.head(tokens[:, 0, :])


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float


class MTNetMethod(BaseMethod):
    method_type = "Fusion"
    name = "MTNet"
    year = "2025"

    def config(self) -> TrainingConfig:
        if self.quick:
            return TrainingConfig(epochs=6, batch_size=256, learning_rate=8e-4, weight_decay=1e-4)
        return TrainingConfig(epochs=26, batch_size=256, learning_rate=6e-4, weight_decay=1e-4)

    def predict(self, ctx: ExperimentContext) -> np.ndarray:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        eeg = ctx.features.raw_eeg.astype(np.float32)
        eye = build_eye_sequence(ctx.features.raw_eye)
        train_eeg, test_eeg = eeg[ctx.train_idx], eeg[ctx.test_idx]
        train_eye, test_eye = eye[ctx.train_idx], eye[ctx.test_idx]

        cfg = self.config()
        model = MTNet(eeg_channels=train_eeg.shape[1], num_classes=3).to(device)
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
