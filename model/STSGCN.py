"""STSGCN EEG graph baseline method.

Implements a spatiotemporal separable graph convolutional network: temporal
gated units extract time features, graph convolution models channel relations,
and separable convolution fuses temporal-spatial representations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from task.train import BaseMethod, ExperimentContext


class TemporalGateUnit(nn.Module):
    def __init__(self, channels: int, hidden: int = 32) -> None:
        super().__init__()
        self.feature_conv = nn.Conv1d(channels, hidden, kernel_size=9, padding=4)
        self.gate_conv = nn.Conv1d(channels, hidden, kernel_size=9, padding=4)
        self.proj = nn.Conv1d(hidden, channels, kernel_size=1)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        feature = torch.tanh(self.feature_conv(eeg))
        gate = torch.sigmoid(self.gate_conv(eeg))
        return self.proj(feature * gate)


class SpatialGraphConvolution(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(node_dim, hidden_dim, bias=False)
        self.key = nn.Linear(node_dim, hidden_dim, bias=False)
        self.value = nn.Linear(node_dim, hidden_dim)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        q = self.query(node_features)
        k = self.key(node_features)
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1])
        adjacency = torch.softmax(scores, dim=-1)
        return F.elu(self.value(torch.matmul(adjacency, node_features)))


class STSGCN(nn.Module):
    def __init__(self, eeg_channels: int, eeg_samples: int, num_classes: int = 3) -> None:
        super().__init__()
        self.temporal_gate = TemporalGateUnit(eeg_channels, hidden=48)
        self.node_projection = nn.Sequential(
            nn.Linear(eeg_samples, 64),
            nn.ELU(),
            nn.Dropout(0.2),
        )
        self.graph = SpatialGraphConvolution(node_dim=64, hidden_dim=64)
        self.separable_fusion = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, padding=2, groups=64, bias=False),
            nn.Conv1d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        temporal = self.temporal_gate(eeg)
        nodes = self.node_projection(temporal)
        nodes = self.graph(nodes)
        fused = self.separable_fusion(nodes.transpose(1, 2)).squeeze(-1)
        return self.head(fused)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float


class STSGCNMethod(BaseMethod):
    method_type = "EEG-only"
    name = "STSGCN"
    year = "2025"

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
        model = STSGCN(train_eeg.shape[1], train_eeg.shape[2], num_classes=3).to(device)
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
