"""Training-side utilities and method registry.

This module owns dataset loading, train/test split construction, feature
extraction, estimator fitting helpers, and the method registry. Concrete
method implementations live under model/, one file per method.
"""

from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence, Tuple, Type

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


class TrainingProgress:
    """Small stderr progress bar for PyTorch training loops."""

    def __init__(self, name: str, epochs: int, steps_per_epoch: int) -> None:
        self.name = name
        self.epochs = int(epochs)
        self.steps_per_epoch = int(steps_per_epoch)
        self.total = self.epochs * self.steps_per_epoch
        self.enabled = self.epochs > 0 and self.steps_per_epoch > 0
        self.current = 0
        self.epoch = 0
        self.last_loss: float | None = None
        self.start_time = 0.0
        self.last_render = 0.0
        self.last_line_len = 0
        self.width = 24

    def __enter__(self) -> "TrainingProgress":
        if self.enabled:
            self.start_time = time.monotonic()
            self._render(force=True)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if not self.enabled:
            return
        should_render = exc_type is not None or self.current < self.total
        if exc_type is None:
            self.current = self.total
            self.epoch = self.epochs
        if should_render:
            self._render(force=True)
        sys.stderr.write("\n")
        sys.stderr.flush()

    def update(self, epoch: int, loss: Any = None) -> None:
        if not self.enabled:
            return
        self.current = min(self.current + 1, self.total)
        self.epoch = epoch
        now = time.monotonic()
        if self.current == self.total or now - self.last_render >= 0.2:
            self._set_loss(loss)
            self._render(now=now)

    def _set_loss(self, loss: Any) -> None:
        if loss is None:
            return
        if hasattr(loss, "detach"):
            loss = loss.detach()
        if hasattr(loss, "item"):
            loss = loss.item()
        self.last_loss = float(loss)

    def _render(self, now: float | None = None, force: bool = False) -> None:
        now = time.monotonic() if now is None else now
        if not force and now - self.last_render < 0.2 and self.current < self.total:
            return
        self.last_render = now

        progress = self.current / self.total
        filled = int(self.width * progress)
        if self.current >= self.total:
            bar = "=" * self.width
        else:
            bar = "=" * filled + ">" + "." * max(self.width - filled - 1, 0)

        elapsed = now - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0.0
        remaining = (self.total - self.current) / rate if rate > 0 else 0.0
        loss_text = f" loss={self.last_loss:.4f}" if self.last_loss is not None else ""
        line = (
            f"[TRAIN] {self.name} epoch {self.epoch}/{self.epochs} "
            f"[{bar}] {self.current}/{self.total} {progress * 100:5.1f}%"
            f"{loss_text} eta={_format_duration(remaining)}"
        )
        padding = " " * max(self.last_line_len - len(line), 0)
        sys.stderr.write("\r" + line + padding)
        sys.stderr.flush()
        self.last_line_len = len(line)


@dataclass(frozen=True)
class Dataset:
    eeg: np.ndarray
    eye: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    subjects: np.ndarray
    tasks: np.ndarray


@dataclass(frozen=True)
class FeatureStore:
    raw_eeg: np.ndarray
    raw_eye: np.ndarray
    eye: np.ndarray
    eye_at_dt: np.ndarray
    eye_at_dt_sequence: np.ndarray
    eye_evidence: np.ndarray
    eeg: np.ndarray
    eeg_graph: np.ndarray
    fusion: np.ndarray
    fusion_graph: np.ndarray

    def get(self, name: str) -> np.ndarray:
        return getattr(self, name)


@dataclass(frozen=True)
class ExperimentContext:
    features: FeatureStore
    train_idx: np.ndarray
    test_idx: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def load_horizon(
    data_dir: Path,
    horizon: int,
    subjects: Sequence[str] | None = None,
    tasks: Sequence[int] | None = None,
) -> Dataset:
    files = sorted((data_dir / f"horizon_{horizon}ms").glob("S*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir / f'horizon_{horizon}ms'}")

    if subjects:
        subject_set = set(subjects)
        files = [path for path in files if path.stem in subject_set]
        found = {path.stem for path in files}
        missing = subject_set - found
        if missing:
            raise FileNotFoundError(f"No processed files found for subject(s): {sorted(missing)}")

    task_set = set(int(task) for task in tasks) if tasks else None
    eeg, eye, y, groups, subject_values, task_values = [], [], [], [], [], []
    for path in files:
        data = np.load(path)
        labels = data["y"].astype(np.int64)
        task_ids = data["task"].astype(np.int64)
        mask = np.ones(len(labels), dtype=bool)
        if task_set is not None:
            mask &= np.isin(task_ids, list(task_set))
        if not np.any(mask):
            continue

        eeg.append(data["eeg"][mask].astype(np.float32))
        eye.append(data["eye"][mask].astype(np.float32))
        labels = labels[mask]
        y.append(labels)
        subj = data["subject"].astype(str)[mask]
        trial = data["trial"].astype(str)[mask]
        task = task_ids[mask].astype(str)
        subject_values.append(subj)
        task_values.append(task_ids[mask])
        groups.append(np.char.add(np.char.add(np.char.add(subj, "_T"), trial), np.char.add("_K", task)))

    if not y:
        raise ValueError(
            f"No samples found in {data_dir / f'horizon_{horizon}ms'} "
            f"after applying subject/task filters."
        )

    return Dataset(
        eeg=np.concatenate(eeg, axis=0),
        eye=np.concatenate(eye, axis=0),
        y=np.concatenate(y, axis=0),
        groups=np.concatenate(groups, axis=0),
        subjects=np.concatenate(subject_values, axis=0),
        tasks=np.concatenate(task_values, axis=0),
    )


def make_split(y: np.ndarray, groups: np.ndarray, seed: int, test_size: float) -> Tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(np.zeros_like(y), y, groups))
    return train_idx, test_idx


def eye_sequence_and_features(eye: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gaze_x = eye[:, :, 0]
    gaze_y = eye[:, :, 1]
    target_x = eye[:, :, 2]
    target_y = eye[:, :, 3]

    dx = (gaze_x - target_x) / 1280.0
    dy = (gaze_y - target_y) / 720.0
    dist_px = np.sqrt((gaze_x - target_x) ** 2 + (gaze_y - target_y) ** 2)
    dist = np.sqrt(dx**2 + dy**2)
    vx = np.diff(gaze_x / 1280.0, axis=1, prepend=gaze_x[:, :1] / 1280.0)
    vy = np.diff(gaze_y / 720.0, axis=1, prepend=gaze_y[:, :1] / 720.0)
    speed = np.sqrt(vx**2 + vy**2)

    seq = np.stack([dx, dy, dist, vx, vy, speed], axis=2).astype(np.float32)
    signals = [dx, dy, dist, dist_px / 1280.0, vx, vy, speed]

    feats = []
    for sig in signals:
        feats.extend(
            [
                sig.mean(axis=1),
                sig.std(axis=1),
                sig.min(axis=1),
                sig.max(axis=1),
                sig[:, 0],
                sig[:, -1],
            ]
        )
    dwell = (dist_px < 70.0).mean(axis=1)
    close_last = (dist_px[:, -1] < 70.0).astype(np.float32)
    feats.extend([dwell, close_last])
    return seq, np.vstack(feats).T.astype(np.float32)


def eye_at_dt_sequence_and_features(eye: np.ndarray, aoi_radius: float = 70.0) -> Tuple[np.ndarray, np.ndarray]:
    gaze_x = eye[:, :, 0]
    gaze_y = eye[:, :, 1]
    target_x = eye[:, :, 2]
    target_y = eye[:, :, 3]

    dist_px = np.sqrt((gaze_x - target_x) ** 2 + (gaze_y - target_y) ** 2)
    a_t = (dist_px < aoi_radius).astype(np.float32)
    d_t = np.clip(dist_px / aoi_radius, 0.0, 5.0).astype(np.float32)

    features = np.column_stack([a_t[:, -1], d_t[:, -1]])
    sequence = np.stack([a_t, d_t], axis=1)
    return sequence.astype(np.float32), features.astype(np.float32)


def eye_evidence_features(eye: np.ndarray) -> np.ndarray:
    gaze_x = eye[:, :, 0]
    gaze_y = eye[:, :, 1]
    target_x = eye[:, :, 2]
    target_y = eye[:, :, 3]
    dist = np.sqrt((gaze_x - target_x) ** 2 + (gaze_y - target_y) ** 2)
    in_aoi = dist < 70.0

    dwell = np.zeros(len(eye), dtype=np.float32)
    for i in range(len(eye)):
        run = 0
        for hit in in_aoi[i, ::-1]:
            if not hit:
                break
            run += 1
        dwell[i] = run / 30.0

    last_dist = dist[:, -1]
    fixation_e = in_aoi[:, -1].astype(np.float32) * (0.6 * dwell + 0.4 * np.clip(1.0 - last_dist / 70.0, 0, 1))
    observation_e = 0.7 * np.clip(last_dist / 70.0, 0, 5)
    evidence = np.stack([fixation_e, observation_e, fixation_e], axis=1)
    strength = evidence.sum(axis=1, keepdims=True)
    uncertainty = 3.0 / (strength + 3.0)
    return np.concatenate([evidence, uncertainty], axis=1).astype(np.float32)


def eeg_features(eeg: np.ndarray, sfreq: float = 250.0) -> np.ndarray:
    # Basic temporal descriptors.
    mean = eeg.mean(axis=2)
    std = eeg.std(axis=2)
    slope = eeg[:, :, -1] - eeg[:, :, 0]
    ptp = eeg.max(axis=2) - eeg.min(axis=2)

    # Band-power descriptors. The input is 0.5 s, so low-frequency resolution is
    # coarse; the first band still captures slow SPN-related deflections.
    freqs = np.fft.rfftfreq(eeg.shape[2], d=1.0 / sfreq)
    spec = np.abs(np.fft.rfft(eeg, axis=2)).astype(np.float32) ** 2
    bands = [(0.1, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 40.0)]
    band_feats = []
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            band_feats.append(np.zeros_like(mean))
        else:
            band_feats.append(np.log1p(spec[:, :, mask].mean(axis=2)))
    del spec
    return np.concatenate([mean, std, slope, ptp] + band_feats, axis=1).astype(np.float32)


def eeg_graph_features(eeg: np.ndarray) -> np.ndarray:
    n, c, t = eeg.shape
    n_regions = 8
    channel_regions = np.array_split(np.arange(c), n_regions)
    region = np.stack([eeg[:, ids, :].mean(axis=1) for ids in channel_regions], axis=1)
    r_mean = region.mean(axis=2)
    r_std = region.std(axis=2)
    centered = region - region.mean(axis=2, keepdims=True)
    denom = np.sqrt(np.sum(centered**2, axis=2, keepdims=True)) + 1e-6
    normed = centered / denom
    corr = np.matmul(normed, np.transpose(normed, (0, 2, 1)))
    iu = np.triu_indices(n_regions, k=1)
    corr_flat = corr[:, iu[0], iu[1]]
    return np.concatenate([r_mean, r_std, corr_flat], axis=1).astype(np.float32)


def fit_predict(estimator, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    model = clone(estimator)
    model.fit(x_train, y_train)
    return model.predict(x_test)


def predict_proba(estimator, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    model = clone(estimator)
    model.fit(x_train, y_train)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)
    decision = model.decision_function(x_test)
    decision = np.atleast_2d(decision)
    decision -= decision.max(axis=1, keepdims=True)
    exp = np.exp(decision)
    return exp / exp.sum(axis=1, keepdims=True)


class BaseMethod(ABC):
    method_type = ""
    name = ""
    year = ""

    def __init__(self, seed: int, quick: bool) -> None:
        self.seed = seed
        self.quick = quick

    @property
    def mlp_iter(self) -> int:
        return 40 if self.quick else 80

    @property
    def tree_n(self) -> int:
        return 120 if self.quick else 240

    def training_progress(self, epochs: int, steps_per_epoch: int) -> TrainingProgress:
        return TrainingProgress(self.name, epochs, steps_per_epoch)

    def x(self, ctx: ExperimentContext, feature_name: str) -> np.ndarray:
        return ctx.features.get(feature_name)

    def fit_predict_estimator(self, estimator, ctx: ExperimentContext, feature_name: str) -> np.ndarray:
        x = self.x(ctx, feature_name)
        return fit_predict(estimator, x[ctx.train_idx], ctx.y_train, x[ctx.test_idx])

    def predict_proba_estimator(self, estimator, ctx: ExperimentContext, feature_name: str) -> np.ndarray:
        x = self.x(ctx, feature_name)
        return predict_proba(estimator, x[ctx.train_idx], ctx.y_train, x[ctx.test_idx])

    @abstractmethod
    def predict(self, ctx: ExperimentContext) -> np.ndarray:
        raise NotImplementedError


class EstimatorMethod(BaseMethod):
    feature_name = ""

    @abstractmethod
    def estimator(self):
        raise NotImplementedError

    def predict(self, ctx: ExperimentContext) -> np.ndarray:
        return self.fit_predict_estimator(self.estimator(), ctx, self.feature_name)


class ProbabilisticFusionMethod(BaseMethod):
    def eye_estimator(self):
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=500, C=1.0, n_jobs=-1, random_state=self.seed),
        )

    def eeg_estimator(self):
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=600, C=1.0, n_jobs=-1, random_state=self.seed),
        )

    def modality_probs(self, ctx: ExperimentContext) -> Tuple[np.ndarray, np.ndarray]:
        p_eye = self.predict_proba_estimator(self.eye_estimator(), ctx, "eye")
        p_eeg = self.predict_proba_estimator(self.eeg_estimator(), ctx, "eeg")
        return p_eye, p_eeg


METHOD_ROWS = [
    ("Eye-only", "SVM", "2014"),
    ("Eye-only", "TCN", "2023"),
    ("EEG-only", "DeepConvNet", "2017"),
    ("EEG-only", "EEGNet", "2018"),
    ("EEG-only", "EEGInception", "2020"),
    ("EEG-only", "MTCN", "2024"),
    ("EEG-only", "PhyTransformer", "2025"),
    ("EEG-only", "DGCN", "2018"),
    ("EEG-only", "STSGCN", "2025"),
    ("Fusion", "MSNet", "2025"),
    ("Fusion", "MTNet", "2025"),
    ("Fusion", "DES", "2026"),
    ("Fusion", "TMVDL", "2022"),
    ("Fusion", "RCML", "2024"),
    ("Fusion", "Ours", "2026"),
]


def method_classes() -> List[Type[BaseMethod]]:
    from model.DES import DESMethod
    from model.DGCN import DGCNMethod
    from model.DeepConvNet import DeepConvNetMethod
    from model.EEGInception import EEGInceptionMethod
    from model.EEGNet import EEGNetMethod
    from model.MSNet import MSNetMethod
    from model.MTCN import MTCNMethod
    from model.MTNet import MTNetMethod
    from model.Ours import OursMethod
    from model.PhyTransformer import PhyTransformerMethod
    from model.RCML import RCMLMethod
    from model.STSGCN import STSGCNMethod
    from model.SVM import SVMMethod
    from model.TCN import TCNMethod
    from model.TMVDL import TMVDLMethod

    return [
        SVMMethod,
        TCNMethod,
        DeepConvNetMethod,
        EEGNetMethod,
        EEGInceptionMethod,
        MTCNMethod,
        PhyTransformerMethod,
        DGCNMethod,
        STSGCNMethod,
        MSNetMethod,
        MTNetMethod,
        DESMethod,
        TMVDLMethod,
        RCMLMethod,
        OursMethod,
    ]


def build_methods(seed: int, quick: bool) -> List[BaseMethod]:
    return [method_cls(seed=seed, quick=quick) for method_cls in method_classes()]
