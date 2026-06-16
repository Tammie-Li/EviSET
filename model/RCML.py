"""RCML reliability-weighted fusion baseline method.

Implements Reliable Conflictive Multi-View Learning with evidential collectors,
average evidence fusion, evidential losses, and a disagreement-confidence loss
for conflictive views.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

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
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RCMLNetwork(nn.Module):
    def __init__(self, eye_dim: int, eeg_dim: int, num_classes: int = 3) -> None:
        super().__init__()
        self.collectors = nn.ModuleDict(
            {
                "eye": EvidenceCollector(eye_dim, num_classes),
                "eeg": EvidenceCollector(eeg_dim, num_classes),
            }
        )

    def forward(self, eye: torch.Tensor, eeg: torch.Tensor):
        evidences: Dict[str, torch.Tensor] = {
            "eye": self.collectors["eye"](eye),
            "eeg": self.collectors["eeg"](eeg),
        }
        evidence_a = (evidences["eye"] + evidences["eeg"]) / 2.0
        return evidences, evidence_a


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


def disagreement_confidence_loss(evidences: Dict[str, torch.Tensor], num_classes: int = 3) -> torch.Tensor:
    ev = list(evidences.values())
    probs = []
    uncertainty = []
    for evidence in ev:
        alpha = evidence + 1.0
        strength = alpha.sum(dim=1, keepdim=True)
        probs.append(alpha / strength)
        uncertainty.append((num_classes / strength).squeeze(1))

    total = torch.zeros((), device=ev[0].device)
    count = 0
    for i in range(len(ev)):
        for j in range(i + 1, len(ev)):
            probability_distance = torch.sum(torch.abs(probs[i] - probs[j]), dim=1) / 2.0
            confidence = (1.0 - uncertainty[i]) * (1.0 - uncertainty[j])
            total = total + torch.mean(probability_distance * confidence)
            count += 1
    return total / max(count, 1)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gamma: float


class RCMLMethod(BaseMethod):
    method_type = "Fusion"
    name = "RCML"
    year = "2024"

    def config(self) -> TrainingConfig:
        if self.quick:
            return TrainingConfig(epochs=8, batch_size=256, learning_rate=1e-3, weight_decay=1e-4, gamma=0.1)
        return TrainingConfig(epochs=35, batch_size=256, learning_rate=8e-4, weight_decay=1e-4, gamma=0.1)

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
        model = RCMLNetwork(eye_train.shape[1], eeg_train.shape[1], num_classes=3).to(device)
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
                    evidences, evidence_a = model(eye_batch, eeg_batch)
                    loss = edl_digamma_loss(evidence_a, y_batch, epoch, 3, cfg.epochs)
                    for evidence in evidences.values():
                        loss = loss + 0.5 * edl_digamma_loss(evidence, y_batch, epoch, 3, cfg.epochs)
                    loss = loss + cfg.gamma * disagreement_confidence_loss(evidences, num_classes=3)
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
                _, evidence = model(eye_batch.to(device), eeg_batch.to(device))
                alpha = evidence + 1.0
                preds.append((alpha / alpha.sum(dim=1, keepdim=True)).argmax(dim=1).cpu().numpy())
        return np.concatenate(preds, axis=0)
