"""TMVDL uncertainty-aware fusion baseline method.

Implements trusted multi-view deep learning with opinion aggregation: each view
learns non-negative evidence, evidence is accumulated across views, and the
Dirichlet opinion is optimized with evidential learning losses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from task.train import BaseMethod, ExperimentContext


class EvidenceCollector(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TMVDLNetwork(nn.Module):
    def __init__(self, eye_dim: int, eeg_dim: int, num_classes: int = 3) -> None:
        super().__init__()
        self.eye_evidence = EvidenceCollector(eye_dim, num_classes)
        self.eeg_evidence = EvidenceCollector(eeg_dim, num_classes)

    def forward(self, eye: torch.Tensor, eeg: torch.Tensor):
        evidence_eye = self.eye_evidence(eye)
        evidence_eeg = self.eeg_evidence(eeg)
        evidence_fused = evidence_eye + evidence_eeg
        return evidence_eye, evidence_eeg, evidence_fused


def kl_divergence(alpha: torch.Tensor, num_classes: int) -> torch.Tensor:
    ones = torch.ones((1, num_classes), dtype=alpha.dtype, device=alpha.device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second = ((alpha - ones) * (torch.digamma(alpha) - torch.digamma(sum_alpha))).sum(dim=1, keepdim=True)
    return first + second


def edl_digamma_loss(
    evidence: torch.Tensor,
    target: torch.Tensor,
    epoch: int,
    num_classes: int,
    annealing_step: int,
) -> torch.Tensor:
    alpha = evidence + 1.0
    y = F.one_hot(target, num_classes=num_classes).float()
    strength = alpha.sum(dim=1, keepdim=True)
    data_fit = torch.sum(y * (torch.digamma(strength) - torch.digamma(alpha)), dim=1, keepdim=True)
    anneal = min(1.0, float(epoch) / float(annealing_step))
    kl_alpha = (alpha - 1.0) * (1.0 - y) + 1.0
    return torch.mean(data_fit + anneal * kl_divergence(kl_alpha, num_classes))


def opinion_entropy(evidence: torch.Tensor) -> torch.Tensor:
    alpha = evidence + 1.0
    probs = alpha / alpha.sum(dim=1, keepdim=True)
    return -torch.sum(probs * torch.log(probs.clamp_min(1e-8)), dim=1)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float


class TMVDLMethod(BaseMethod):
    method_type = "Fusion"
    name = "TMVDL"
    year = "2022"

    def config(self) -> TrainingConfig:
        if self.quick:
            return TrainingConfig(epochs=8, batch_size=256, learning_rate=1e-3, weight_decay=1e-4)
        return TrainingConfig(epochs=35, batch_size=256, learning_rate=8e-4, weight_decay=1e-4)

    def predict(self, ctx: ExperimentContext) -> np.ndarray:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        eye_scaler = StandardScaler()
        eeg_scaler = StandardScaler()
        eye_train = eye_scaler.fit_transform(ctx.features.eye[ctx.train_idx]).astype(np.float32)
        eye_test = eye_scaler.transform(ctx.features.eye[ctx.test_idx]).astype(np.float32)
        eeg_train = eeg_scaler.fit_transform(ctx.features.eeg[ctx.train_idx]).astype(np.float32)
        eeg_test = eeg_scaler.transform(ctx.features.eeg[ctx.test_idx]).astype(np.float32)

        cfg = self.config()
        model = TMVDLNetwork(eye_train.shape[1], eeg_train.shape[1], num_classes=3).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        train_ds = TensorDataset(
            torch.from_numpy(eye_train),
            torch.from_numpy(eeg_train),
            torch.from_numpy(ctx.y_train.astype(np.int64)),
        )
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, generator=generator)

        model.train()
        with self.training_progress(cfg.epochs, len(train_loader)) as progress:
            for epoch in range(1, cfg.epochs + 1):
                for eye_batch, eeg_batch, y_batch in train_loader:
                    eye_batch = eye_batch.to(device)
                    eeg_batch = eeg_batch.to(device)
                    y_batch = y_batch.to(device)
                    evidence_eye, evidence_eeg, evidence_fused = model(eye_batch, eeg_batch)
                    loss = edl_digamma_loss(evidence_fused, y_batch, epoch, 3, cfg.epochs)
                    loss = loss + 0.5 * edl_digamma_loss(evidence_eye, y_batch, epoch, 3, cfg.epochs)
                    loss = loss + 0.5 * edl_digamma_loss(evidence_eeg, y_batch, epoch, 3, cfg.epochs)
                    consistency = torch.mean(torch.abs(opinion_entropy(evidence_eye) - opinion_entropy(evidence_eeg)))
                    loss = loss + 0.05 * consistency
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    progress.update(epoch, loss.detach())

        model.eval()
        preds = []
        test_ds = TensorDataset(torch.from_numpy(eye_test), torch.from_numpy(eeg_test))
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)
        with torch.no_grad():
            for eye_batch, eeg_batch in test_loader:
                _, _, evidence = model(eye_batch.to(device), eeg_batch.to(device))
                alpha = evidence + 1.0
                preds.append((alpha / alpha.sum(dim=1, keepdim=True)).argmax(dim=1).cpu().numpy())
        return np.concatenate(preds, axis=0)
