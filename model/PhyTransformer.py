"""PhyTransformer EEG-only baseline method.

This implementation follows the PhyTransformer principle of extracting local
temporal physiological patterns and modeling global channel relations with a
Transformer encoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from task.train import BaseMethod, ExperimentContext


class ChannelTemporalEmbedding(nn.Module):
    def __init__(self, d_model: int = 48) -> None:
        super().__init__()
        self.local_encoder = nn.Sequential(
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
        x = self.local_encoder(x).squeeze(-1)
        return x.reshape(batch, channels, -1)


class PhyTransformer(nn.Module):
    def __init__(
        self,
        eeg_channels: int,
        num_classes: int = 3,
        d_model: int = 48,
        nhead: int = 4,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.embedding = ChannelTemporalEmbedding(d_model=d_model)
        self.channel_pos = nn.Parameter(torch.zeros(1, eeg_channels, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.25,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        tokens = self.embedding(eeg) + self.channel_pos
        tokens = self.transformer(tokens)
        return self.head(tokens.mean(dim=1))


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float


class PhyTransformerMethod(BaseMethod):
    method_type = "EEG-only"
    name = "PhyTransformer"
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
        train_eeg = eeg[ctx.train_idx]
        test_eeg = eeg[ctx.test_idx]

        cfg = self.config()
        model = PhyTransformer(eeg_channels=train_eeg.shape[1], num_classes=3).to(device)
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
