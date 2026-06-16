"""Preprocess EviSET raw recordings into three early-prediction datasets.

The raw directory is expected to contain one directory per subject:

    raw_data/S01/
        Neuracle/data.bdf
        Neuracle/evt.bdf
        target_loc.npy
        trial_data/*_gaze_samples.csv

For every valid task triple in evt.bdf (1, 2, 3), this script extracts the
500-ms EEG and eye-movement window preceding each trigger at three prediction
horizons:

    500 ms ahead: [-1000, -500] ms relative to the trigger
    250 ms ahead: [ -750, -250] ms relative to the trigger
      0 ms ahead: [ -500,    0] ms relative to the trigger

EEG preprocessing is applied before epoch extraction:
    1. keep scalp EEG channels and exclude auxiliary ECG/EOG channels;
    2. average reference;
    3. downsampling to 250 Hz by default;
    4. 0.1-48 Hz 6th-order Butterworth band-pass filtering;
    5. channel-wise z-score normalization within each subject recording.

Labels:
    0: fixation without selection intention, preceding Trigger 1
    1: observation, preceding Trigger 2
    2: fixation with selection intention, preceding Trigger 3
"""

from __future__ import annotations

import argparse
import csv
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import mne
import numpy as np
import pandas as pd


HORIZONS: Dict[int, Tuple[float, float]] = {
    500: (-1.00, -0.50),
    250: (-0.75, -0.25),
    0: (-0.50, 0.00),
}

TRIGGER_TO_LABEL = {"1": 0, "2": 1, "3": 2}
AUX_CHANNELS = {"ECG", "HEOR", "HEOL", "VEOU", "VEOL"}


@dataclass(frozen=True)
class EventSample:
    subject: str
    trial: int
    task: int
    trigger: int
    label: int
    eeg_time: float
    gaze_time: float
    target_xy: Tuple[float, float]
    is_instruction: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("raw_data"))
    parser.add_argument("--out-dir", type=Path, default=Path("processed_data"))
    parser.add_argument("--subjects", "--subject", dest="subjects", nargs="*", default=None)
    parser.add_argument(
        "--tasks",
        "--task",
        dest="tasks",
        nargs="*",
        type=int,
        default=None,
        help="Optional task slot ids to preprocess, for example: --task 1 3 5",
    )
    parser.add_argument("--sfreq", type=float, default=250.0)
    parser.add_argument("--eye-steps", type=int, default=30)
    parser.add_argument("--l-freq", type=float, default=0.1)
    parser.add_argument("--h-freq", type=float, default=48.0)
    parser.add_argument("--filter-order", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Skip EEG band-pass filtering. Useful for quick smoke tests.",
    )
    parser.add_argument(
        "--no-zscore",
        action="store_true",
        help="Skip channel-wise z-score normalization.",
    )
    parser.add_argument(
        "--include-aux-channels",
        action="store_true",
        help="Keep auxiliary ECG/EOG channels in the saved EEG array. By default they are excluded.",
    )
    return parser.parse_args()


def subject_dirs(raw_dir: Path, subjects: Sequence[str] | None) -> List[Path]:
    if subjects:
        dirs = [raw_dir / s for s in subjects]
    else:
        dirs = sorted(p for p in raw_dir.iterdir() if p.is_dir() and p.name.startswith("S"))
    missing = [str(p) for p in dirs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing subject directories: {missing}")
    return dirs


def read_annotations(evt_path: Path) -> List[Tuple[float, str]]:
    raw_evt = mne.io.read_raw_bdf(str(evt_path), preload=False, verbose="ERROR")
    return [
        (float(onset), str(desc))
        for onset, desc in zip(raw_evt.annotations.onset, raw_evt.annotations.description)
    ]


def load_gaze_trials(trial_dir: Path) -> List[pd.DataFrame]:
    files = sorted(trial_dir.glob("*_gaze_samples.csv"))
    trials: List[pd.DataFrame] = []
    for path in files:
        df = pd.read_csv(path)
        required = {"time", "x", "y", "trigger"}
        if not required.issubset(df.columns):
            raise ValueError(f"{path} lacks required columns {required}")
        trials.append(df)
    return trials


def trial_start_time(df: pd.DataFrame) -> float:
    starts = df.loc[df["trigger"].astype(int) == 101, "time"].to_numpy()
    if len(starts) == 0:
        return float(df["time"].iloc[0])
    return float(starts[0])


def build_event_samples(subject_dir: Path, gaze_trials: List[pd.DataFrame]) -> List[EventSample]:
    subject = subject_dir.name
    annotations = read_annotations(subject_dir / "Neuracle" / "evt.bdf")
    target_locs = np.load(subject_dir / "target_loc.npy").reshape(100, 5, 2)

    starts = [i for i, (_, desc) in enumerate(annotations) if desc == "101"]
    if len(starts) != len(gaze_trials):
        print(
            f"[WARN] {subject}: {len(starts)} EEG trial starts vs "
            f"{len(gaze_trials)} gaze files; using the common prefix."
        )

    n_trials = min(len(starts), len(gaze_trials), target_locs.shape[0])
    samples: List[EventSample] = []

    for trial_idx in range(n_trials):
        start_i = starts[trial_idx]
        end_i = starts[trial_idx + 1] if trial_idx + 1 < len(starts) else len(annotations)
        trial_ann = annotations[start_i + 1 : end_i]

        # Stop at the end-of-trial marker and keep only task markers.
        clean_ann: List[Tuple[float, str]] = []
        for onset, desc in trial_ann:
            if desc == "102":
                break
            if desc in {"0", "1", "2", "3"}:
                clean_ann.append((onset, desc))

        eeg_trial_start = annotations[start_i][0]
        gaze_start = trial_start_time(gaze_trials[trial_idx])

        # The experiment contains five task slots per trial. Invalid tasks are
        # represented by 0,0,0 in evt.bdf and still consume one target_loc row.
        for task_idx in range(min(5, len(clean_ann) // 3)):
            triple = clean_ann[3 * task_idx : 3 * task_idx + 3]
            descs = [d for _, d in triple]
            if descs != ["1", "2", "3"]:
                continue

            target_xy = tuple(float(v) for v in target_locs[trial_idx, task_idx])
            for onset, desc in triple:
                label = TRIGGER_TO_LABEL[desc]
                gaze_time = gaze_start + (float(onset) - float(eeg_trial_start))
                samples.append(
                    EventSample(
                        subject=subject,
                        trial=trial_idx + 1,
                        task=task_idx + 1,
                        trigger=int(desc),
                        label=label,
                        eeg_time=float(onset),
                        gaze_time=float(gaze_time),
                        target_xy=target_xy,
                        is_instruction=(desc == "1"),
                    )
                )

    return samples


def estimate_instruction_xy(samples: Sequence[EventSample], gaze_trials: Sequence[pd.DataFrame]) -> Tuple[float, float]:
    points: List[np.ndarray] = []
    for sample in samples:
        if not sample.is_instruction:
            continue
        df = gaze_trials[sample.trial - 1]
        segment = df[(df["time"] >= sample.gaze_time - 0.20) & (df["time"] < sample.gaze_time)]
        if len(segment) > 0:
            points.append(segment[["x", "y"]].median().to_numpy(dtype=float))
    if points:
        xy = np.median(np.vstack(points), axis=0)
        return float(xy[0]), float(xy[1])

    trigger_points = []
    for df in gaze_trials:
        hit = df[df["trigger"].astype(int) == 1]
        if len(hit) > 0:
            trigger_points.append(hit[["x", "y"]].median().to_numpy(dtype=float))
    if trigger_points:
        xy = np.median(np.vstack(trigger_points), axis=0)
        return float(xy[0]), float(xy[1])
    return 640.0, 100.0


def sample_instruction_xy(
    sample: EventSample,
    df: pd.DataFrame,
    fallback_xy: Tuple[float, float],
    lookback: float = 0.20,
) -> Tuple[float, float]:
    segment = df[(df["time"] >= sample.gaze_time - lookback) & (df["time"] < sample.gaze_time)]
    if len(segment) == 0:
        return fallback_xy
    xy = segment[["x", "y"]].median().to_numpy(dtype=float)
    return float(xy[0]), float(xy[1])


def resample_eye_window(
    df: pd.DataFrame,
    start_time: float,
    stop_time: float,
    target_xy: Tuple[float, float],
    n_steps: int,
) -> np.ndarray:
    times = np.linspace(start_time, stop_time, n_steps, endpoint=False, dtype=np.float64)
    src_t = df["time"].to_numpy(dtype=np.float64)
    src_x = df["x"].to_numpy(dtype=np.float64)
    src_y = df["y"].to_numpy(dtype=np.float64)

    gaze_x = np.interp(times, src_t, src_x, left=src_x[0], right=src_x[-1])
    gaze_y = np.interp(times, src_t, src_y, left=src_y[0], right=src_y[-1])
    target_x = np.full_like(gaze_x, float(target_xy[0]))
    target_y = np.full_like(gaze_y, float(target_xy[1]))
    return np.stack([gaze_x, gaze_y, target_x, target_y], axis=1).astype(np.float32)


def zscore_raw(raw: mne.io.BaseRaw) -> None:
    data = raw.get_data()
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    std[std < 1e-8] = 1.0
    raw._data = ((data - mean) / std).astype(np.float64, copy=False)


def read_eeg(
    data_path: Path,
    sfreq: float,
    l_freq: float,
    h_freq: float,
    filter_order: int,
    no_filter: bool,
    no_zscore: bool,
    include_aux_channels: bool,
) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_bdf(str(data_path), preload=True, verbose="ERROR")
    if not include_aux_channels:
        eeg_names = [ch for ch in raw.ch_names if ch.upper() not in AUX_CHANNELS]
        dropped = [ch for ch in raw.ch_names if ch.upper() in AUX_CHANNELS]
        raw.pick(eeg_names)
        if dropped:
            print(f"[INFO] excluded auxiliary channels: {', '.join(dropped)}")
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    raw.resample(sfreq, npad="auto", verbose="ERROR")
    if not no_filter:
        iir_params = dict(order=filter_order, ftype="butter")
        raw.filter(
            l_freq=l_freq,
            h_freq=h_freq,
            method="iir",
            iir_params=iir_params,
            verbose="ERROR",
        )
    if not no_zscore:
        zscore_raw(raw)
    return raw


def extract_eeg_window(raw: mne.io.BaseRaw, start_time: float, stop_time: float) -> np.ndarray | None:
    sfreq = float(raw.info["sfreq"])
    n_samples = int(round((stop_time - start_time) * sfreq))
    start = int(round(start_time * sfreq))
    stop = start + n_samples
    if start < 0 or stop > raw.n_times:
        return None
    data = raw.get_data(start=start, stop=stop).astype(np.float32)
    if data.shape[1] != n_samples:
        return None
    return data


def ensure_output_dirs(out_dir: Path, horizons: Iterable[int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for horizon in horizons:
        (out_dir / f"horizon_{horizon}ms").mkdir(parents=True, exist_ok=True)


def save_subject_horizon(
    out_path: Path,
    eeg: List[np.ndarray],
    eye: List[np.ndarray],
    labels: List[int],
    rows: List[dict],
    eeg_channels: Sequence[str],
) -> None:
    np.savez(
        out_path,
        eeg=np.stack(eeg).astype(np.float32),
        eeg_channels=np.asarray(eeg_channels),
        eye=np.stack(eye).astype(np.float32),
        y=np.asarray(labels, dtype=np.int64),
        subject=np.asarray([r["subject"] for r in rows]),
        trial=np.asarray([r["trial"] for r in rows], dtype=np.int64),
        task=np.asarray([r["task"] for r in rows], dtype=np.int64),
        trigger=np.asarray([r["trigger"] for r in rows], dtype=np.int64),
        event_time=np.asarray([r["event_time"] for r in rows], dtype=np.float64),
        target_xy=np.asarray([r["target_xy"] for r in rows], dtype=np.float32),
    )


def preprocess_subject(args: argparse.Namespace, subject_dir: Path) -> List[dict]:
    subject = subject_dir.name
    out_paths = {
        horizon: args.out_dir / f"horizon_{horizon}ms" / f"{subject}.npz"
        for horizon in HORIZONS
    }
    if all(p.exists() for p in out_paths.values()) and not args.overwrite:
        print(f"[SKIP] {subject}: processed files already exist")
        return []

    print(f"[INFO] {subject}: loading gaze/events")
    gaze_trials = load_gaze_trials(subject_dir / "trial_data")
    samples = build_event_samples(subject_dir, gaze_trials)
    if args.tasks:
        task_set = {int(task) for task in args.tasks}
        samples = [sample for sample in samples if sample.task in task_set]
    else:
        task_set = None
    instruction_xy = estimate_instruction_xy(samples, gaze_trials)
    task_text = f", tasks={sorted(task_set)}" if task_set is not None else ""
    print(
        f"[INFO] {subject}: {len(samples)} trigger samples{task_text}, "
        f"instruction_xy=({instruction_xy[0]:.1f}, {instruction_xy[1]:.1f})"
    )
    if not samples:
        print(f"[WARN] {subject}: no trigger samples after filtering")
        return []

    print(f"[INFO] {subject}: loading and preprocessing EEG")
    raw = read_eeg(
        subject_dir / "Neuracle" / "data.bdf",
        sfreq=args.sfreq,
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        filter_order=args.filter_order,
        no_filter=args.no_filter,
        no_zscore=args.no_zscore,
        include_aux_channels=args.include_aux_channels,
    )

    horizon_data = {
        horizon: {"eeg": [], "eye": [], "labels": [], "rows": []}
        for horizon in HORIZONS
    }
    manifest_rows: List[dict] = []

    for sample in samples:
        df = gaze_trials[sample.trial - 1]
        aoi_xy = sample_instruction_xy(sample, df, instruction_xy) if sample.is_instruction else sample.target_xy

        for horizon, (offset_start, offset_stop) in HORIZONS.items():
            eeg_start = sample.eeg_time + offset_start
            eeg_stop = sample.eeg_time + offset_stop
            eye_start = sample.gaze_time + offset_start
            eye_stop = sample.gaze_time + offset_stop

            eeg_window = extract_eeg_window(raw, eeg_start, eeg_stop)
            if eeg_window is None:
                continue
            eye_window = resample_eye_window(df, eye_start, eye_stop, aoi_xy, args.eye_steps)

            row = {
                "subject": subject,
                "trial": sample.trial,
                "task": sample.task,
                "trigger": sample.trigger,
                "label": sample.label,
                "horizon_ms": horizon,
                "event_time": sample.eeg_time,
                "target_xy": sample.target_xy,
            }
            bucket = horizon_data[horizon]
            bucket["eeg"].append(eeg_window)
            bucket["eye"].append(eye_window)
            bucket["labels"].append(sample.label)
            bucket["rows"].append(row)
            manifest_rows.append(row)

    for horizon, bucket in horizon_data.items():
        out_path = out_paths[horizon]
        if len(bucket["labels"]) == 0:
            print(f"[WARN] {subject}: no samples for horizon {horizon}ms")
            continue
        save_subject_horizon(
            out_path,
            bucket["eeg"],
            bucket["eye"],
            bucket["labels"],
            bucket["rows"],
            raw.ch_names,
        )
        counts = np.bincount(np.asarray(bucket["labels"]), minlength=3)
        print(f"[OK] {subject}: {out_path} samples={len(bucket['labels'])} labels={counts.tolist()}")

    del raw
    gc.collect()
    return manifest_rows


def write_manifest(out_dir: Path, rows: Sequence[dict], append: bool) -> None:
    path = out_dir / "manifest.csv"
    fieldnames = ["subject", "trial", "task", "trigger", "label", "horizon_ms", "event_time", "target_x", "target_y"]
    mode = "a" if append and path.exists() else "w"
    with path.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        for row in rows:
            target_x, target_y = row["target_xy"]
            writer.writerow(
                {
                    "subject": row["subject"],
                    "trial": row["trial"],
                    "task": row["task"],
                    "trigger": row["trigger"],
                    "label": row["label"],
                    "horizon_ms": row["horizon_ms"],
                    "event_time": f"{row['event_time']:.6f}",
                    "target_x": f"{float(target_x):.6f}",
                    "target_y": f"{float(target_y):.6f}",
                }
            )


def main() -> None:
    args = parse_args()
    ensure_output_dirs(args.out_dir, HORIZONS)

    all_rows: List[dict] = []
    for i, subject_dir in enumerate(subject_dirs(args.raw_dir, args.subjects)):
        rows = preprocess_subject(args, subject_dir)
        if rows:
            write_manifest(args.out_dir, rows, append=(i > 0 and not args.overwrite))
            all_rows.extend(rows)

    if all_rows:
        counts = {}
        for horizon in HORIZONS:
            labels = [r["label"] for r in all_rows if r["horizon_ms"] == horizon]
            counts[horizon] = np.bincount(np.asarray(labels), minlength=3).tolist()
        print(f"[DONE] Wrote {len(all_rows)} window samples. Label counts by horizon: {counts}")
    else:
        print("[DONE] No new samples were written.")


if __name__ == "__main__":
    main()
