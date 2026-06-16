"""DGCN EEG graph baseline method.

Implements a dynamical graph convolutional network for multichannel EEG. Channel
nodes are represented by temporal embeddings and a sample-adaptive adjacency
matrix is learned from node features together with a trainable graph prior.
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


class NodeTemporalEncoder(nn.Module):
    def __init__(self, node_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, padding=5, bias=False),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.Conv1d(16, node_dim, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(node_dim),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        batch, channels, samples = eeg.shape
        x = eeg.reshape(batch * channels, 1, samples)
        x = self.net(x).squeeze(-1)
        return x.reshape(batch, channels, -1)


class DynamicGraphConvolution(nn.Module):
    def __init__(self, channels: int, node_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.static_adj = nn.Parameter(torch.eye(channels))
        self.query = nn.Linear(node_dim, hidden_dim, bias=False)
        self.key = nn.Linear(node_dim, hidden_dim, bias=False)
        self.value = nn.Linear(node_dim, hidden_dim)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        q = self.query(nodes)
        k = self.key(nodes)
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1])
        scores = scores + self.static_adj.unsqueeze(0)
        adjacency = torch.softmax(scores, dim=-1)
        aggregated = torch.matmul(adjacency, nodes)
        return F.elu(self.value(aggregated))


class DGCN(nn.Module):
    def __init__(self, eeg_channels: int, num_classes: int = 3) -> None:
        super().__init__()
        self.node_encoder = NodeTemporalEncoder(node_dim=32)
        self.graph1 = DynamicGraphConvolution(eeg_channels, node_dim=32, hidden_dim=48)
        self.graph2 = DynamicGraphConvolution(eeg_channels, node_dim=48, hidden_dim=48)
        self.head = nn.Sequential(
            nn.LayerNorm(48),
            nn.Linear(48, 64),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        nodes = self.node_encoder(eeg)
        nodes = self.graph1(nodes)
        nodes = self.graph2(nodes)
        return self.head(nodes.mean(dim=1))


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float


class DGCNMethod(BaseMethod):
    method_type = "EEG-only"
    name = "DGCN"
    year = "2018"

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
        model = DGCN(eeg_channels=train_eeg.shape[1], num_classes=3).to(device)
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
