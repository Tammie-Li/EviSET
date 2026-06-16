"""EviSET evidence/opinion fusion method.

All computations specific to the proposed method are kept in this file:

1. eye-movement evidence accumulation from raw gaze/AOI trajectories;
2. multi-scale temporal convolution over raw EEG windows;
3. eye-evidence-guided temporal-scale reorganization;
4. sample-adaptive dynamic graph convolution over EEG channels;
5. evidence-to-opinion mapping and reliability-aware opinion fusion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from task.train import BaseMethod, ExperimentContext, fit_predict


def eye_movement_evidence(
    eye_at_dt_sequence: np.ndarray,
    aoi_radius: float = 70.0,
    dwell_threshold: float = 0.75,
    lambda_fix_time: float = 0.6,
    lambda_fix_dist: float = 0.4,
    lambda_obs_dist: float = 0.7,
) -> np.ndarray:
    """Compute deterministic eye-movement evidence from At/dt windows.

    Input shape is (N, 2, T), with channels [A_t, d_t].
    Output shape is (N, 3), ordered as:
    fixation without intention, observation, fixation with intention.
    """

    a_t = eye_at_dt_sequence[:, 0, :].astype(np.float32)
    d_t = eye_at_dt_sequence[:, 1, :].astype(np.float32)

    dt = 1.0 / 60.0
    dwell = np.zeros(len(eye_at_dt_sequence), dtype=np.float32)
    for i in range(len(eye_at_dt_sequence)):
        continuous_steps = 0
        for hit in a_t[i, ::-1]:
            if not hit:
                break
            continuous_steps += 1
        dwell[i] = continuous_steps * dt

    last_hit = a_t[:, -1].astype(np.float32)
    last_distance = d_t[:, -1] * aoi_radius
    fixation_evidence = last_hit * (
        lambda_fix_time * np.clip(dwell / dwell_threshold, 0.0, 1.0)
        + lambda_fix_dist * np.clip(1.0 - last_distance / aoi_radius, 0.0, 1.0)
    )
    observation_evidence = lambda_obs_dist * np.clip(last_distance / aoi_radius, 0.0, 5.0)

    evidence = np.stack([fixation_evidence, observation_evidence, fixation_evidence], axis=1)
    return evidence.astype(np.float32)


def evidence_to_opinion(evidence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map non-negative evidence to belief, uncertainty, and predictive score."""

    num_classes = evidence.shape[1]
    strength = evidence.sum(dim=1, keepdim=True) + float(num_classes)
    belief = evidence / strength
    uncertainty = float(num_classes) / strength
    probability = belief + uncertainty / float(num_classes)
    return belief, uncertainty, probability


def reliability_aware_fusion(eye_evidence: torch.Tensor, eeg_evidence: torch.Tensor) -> torch.Tensor:
    """Fuse eye and EEG evidence through uncertainty-derived reliability."""

    _, eye_uncertainty, _ = evidence_to_opinion(eye_evidence)
    _, eeg_uncertainty, _ = evidence_to_opinion(eeg_evidence)
    eye_reliability = 1.0 - eye_uncertainty
    eeg_reliability = 1.0 - eeg_uncertainty
    denom = eye_reliability + eeg_reliability + 1e-6
    eye_weight = eye_reliability / denom
    eeg_weight = eeg_reliability / denom
    return eye_weight * eye_evidence + eeg_weight * eeg_evidence


class SameLengthDepthwiseTemporalConv(nn.Module):
    """Depthwise temporal convolution preserving channel nodes."""

    def __init__(self, channels: int, features_per_channel: int, kernel_size: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels * features_per_channel,
            kernel_size=kernel_size,
            groups=channels,
            padding=kernel_size // 2,
        )
        self.channels = channels
        self.features_per_channel = features_per_channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        target_len = x.shape[-1]
        if y.shape[-1] > target_len:
            y = y[:, :, :target_len]
        elif y.shape[-1] < target_len:
            y = nn.functional.pad(y, (0, target_len - y.shape[-1]))
        y = nn.functional.elu(y)
        y = y.view(x.shape[0], self.channels, self.features_per_channel, target_len)
        return y.mean(dim=-1)


class DynamicGraphConvolution(nn.Module):
    """Sample-adaptive graph convolution across EEG channels."""

    def __init__(self, node_dim: int, graph_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(node_dim, graph_dim, bias=False)
        self.key = nn.Linear(node_dim, graph_dim, bias=False)
        self.value = nn.Linear(node_dim, graph_dim)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        q = self.query(node_features)
        k = self.key(node_features)
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1])
        adjacency = torch.softmax(scores, dim=-1)
        aggregated = torch.matmul(adjacency, node_features)
        return nn.functional.elu(self.value(aggregated))


class EviSETNetwork(nn.Module):
    """EEG evidence learner plus evidence/opinion fusion."""

    def __init__(
        self,
        eeg_channels: int = 64,
        node_dim: int = 4,
        graph_dim: int = 32,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.temporal_branches = nn.ModuleList(
            [
                SameLengthDepthwiseTemporalConv(eeg_channels, node_dim, kernel_size=8),
                SameLengthDepthwiseTemporalConv(eeg_channels, node_dim, kernel_size=16),
                SameLengthDepthwiseTemporalConv(eeg_channels, node_dim, kernel_size=32),
            ]
        )
        self.eye_scale_gate = nn.Sequential(nn.Linear(num_classes, 16), nn.ELU(), nn.Linear(16, 3))
        self.graph_conv = DynamicGraphConvolution(node_dim=node_dim, graph_dim=graph_dim)
        self.eeg_head = nn.Sequential(
            nn.Linear(graph_dim, 32),
            nn.ELU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, num_classes),
            nn.Softplus(),
        )

    def forward(self, eeg: torch.Tensor, eye_evidence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        branch_features = [branch(eeg) for branch in self.temporal_branches]
        scale_weight = torch.softmax(self.eye_scale_gate(eye_evidence), dim=1)
        node_features = sum(
            scale_weight[:, i].view(-1, 1, 1) * branch_features[i]
            for i in range(len(branch_features))
        )

        graph_features = self.graph_conv(node_features)
        eeg_summary = graph_features.mean(dim=1)
        eeg_evidence = self.eeg_head(eeg_summary)
        fused_evidence = reliability_aware_fusion(eye_evidence, eeg_evidence)
        _, _, fused_probability = evidence_to_opinion(fused_evidence)
        _, _, eeg_probability = evidence_to_opinion(eeg_evidence)
        _, _, eye_probability = evidence_to_opinion(eye_evidence)
        return fused_probability, eeg_probability, eye_probability


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float


class OursMethod(BaseMethod):
    method_type = "Fusion"
    name = "Ours"
    year = "2026"

    def config(self) -> TrainingConfig:
        if self.quick:
            return TrainingConfig(epochs=6, batch_size=256, learning_rate=1e-3, weight_decay=1e-4)
        return TrainingConfig(epochs=18, batch_size=256, learning_rate=8e-4, weight_decay=1e-4)

    def predict(self, ctx: ExperimentContext) -> np.ndarray:
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=900, C=1.5, n_jobs=-1, random_state=self.seed),
        )
        x = ctx.features.fusion_graph
        return fit_predict(estimator, x[ctx.train_idx], ctx.y_train, x[ctx.test_idx])

    def predict_deep(self, ctx: ExperimentContext) -> np.ndarray:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_eeg = ctx.features.raw_eeg[ctx.train_idx]
        test_eeg = ctx.features.raw_eeg[ctx.test_idx]
        train_eye_ev = eye_movement_evidence(ctx.features.eye_at_dt_sequence[ctx.train_idx])
        test_eye_ev = eye_movement_evidence(ctx.features.eye_at_dt_sequence[ctx.test_idx])

        model = EviSETNetwork(eeg_channels=train_eeg.shape[1]).to(device)
        cfg = self.config()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        train_ds = TensorDataset(
            torch.from_numpy(train_eeg.astype(np.float32)),
            torch.from_numpy(train_eye_ev.astype(np.float32)),
            torch.from_numpy(ctx.y_train.astype(np.int64)),
        )
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, generator=generator)

        model.train()
        with self.training_progress(cfg.epochs, len(loader)) as progress:
            for epoch in range(1, cfg.epochs + 1):
                for eeg_batch, eye_ev_batch, y_batch in loader:
                    eeg_batch = eeg_batch.to(device)
                    eye_ev_batch = eye_ev_batch.to(device)
                    y_batch = y_batch.to(device)

                    fused_prob, eeg_prob, _ = model(eeg_batch, eye_ev_batch)
                    loss_fused = nn.functional.nll_loss(torch.log(fused_prob.clamp_min(1e-8)), y_batch)
                    loss_eeg = nn.functional.nll_loss(torch.log(eeg_prob.clamp_min(1e-8)), y_batch)
                    loss = loss_fused + 0.2 * loss_eeg

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    progress.update(epoch, loss.detach())

        model.eval()
        preds = []
        test_ds = TensorDataset(
            torch.from_numpy(test_eeg.astype(np.float32)),
            torch.from_numpy(test_eye_ev.astype(np.float32)),
        )
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
        with torch.no_grad():
            for eeg_batch, eye_ev_batch in test_loader:
                fused_prob, _, _ = model(eeg_batch.to(device), eye_ev_batch.to(device))
                preds.append(fused_prob.argmax(dim=1).cpu().numpy())
        return np.concatenate(preds, axis=0)
