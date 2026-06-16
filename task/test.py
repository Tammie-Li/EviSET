"""Experiment and evaluation runner.

This module owns the evaluation workflow:
load one prediction-horizon dataset, build features, run all method classes,
compute metrics, and write subject-aggregated CSV summaries for
within-subject and cross-subject protocols.
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import fields
from pathlib import Path
from typing import List, Sequence

import numpy as np

from task.metric import evaluate_predictions, write_outputs
from task.train import (
    ExperimentContext,
    FeatureStore,
    METHOD_ROWS,
    build_methods,
    eeg_features,
    eeg_graph_features,
    eye_at_dt_sequence_and_features,
    eye_evidence_features,
    load_horizon,
    make_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("processed_data"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--protocol",
        choices=["within", "cross", "both", "pooled"],
        default="both",
        help=(
            "Evaluation protocol. 'within' splits each subject internally; "
            "'cross' uses leave-one-subject-out; 'both' writes both summaries; "
            "'pooled' reproduces the old mixed-subject split."
        ),
    )
    parser.add_argument(
        "--methods",
        "--models",
        "--model",
        dest="methods",
        nargs="+",
        default=None,
        help="Optional model/method names to run, for example: --model SVM Ours",
    )
    parser.add_argument(
        "--subjects",
        "--subject",
        dest="subjects",
        nargs="*",
        default=None,
        help="Optional subject ids to evaluate, for example: --subject S01 S02",
    )
    parser.add_argument(
        "--tasks",
        "--task",
        dest="tasks",
        nargs="*",
        type=int,
        default=None,
        help="Optional task slot ids to evaluate, for example: --task 1 3 5",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use faster estimators. Useful for smoke tests; not recommended for final numbers.",
    )
    return parser.parse_args()


def requested_protocols(protocol: str) -> List[str]:
    if protocol == "both":
        return ["within_subject", "cross_subject"]
    if protocol == "within":
        return ["within_subject"]
    if protocol == "cross":
        return ["cross_subject"]
    return ["pooled"]


def selected_method_rows(method_names: Sequence[str] | None) -> List[tuple]:
    if method_names is None:
        return METHOD_ROWS
    wanted = set(method_names)
    rows = [row for row in METHOD_ROWS if row[1] in wanted]
    missing = wanted - {row[1] for row in rows}
    if missing:
        raise ValueError(f"Unknown method(s): {', '.join(sorted(missing))}")
    return rows


def build_selected_methods(seed: int, quick: bool, method_names: Sequence[str] | None):
    methods = build_methods(seed, quick)
    if method_names is None:
        return methods
    wanted = set(method_names)
    selected = [method_impl for method_impl in methods if method_impl.name in wanted]
    missing = wanted - {method_impl.name for method_impl in selected}
    if missing:
        raise ValueError(f"Unknown method(s): {', '.join(sorted(missing))}")
    return selected


def build_feature_store(dataset) -> FeatureStore:
    x_eye_at_dt_seq, x_eye_at_dt = eye_at_dt_sequence_and_features(dataset.eye)
    x_eye_ev = eye_evidence_features(dataset.eye)
    x_eeg = eeg_features(dataset.eeg)
    x_graph = eeg_graph_features(dataset.eeg)
    return FeatureStore(
        raw_eeg=dataset.eeg,
        raw_eye=dataset.eye,
        eye=x_eye_at_dt,
        eye_at_dt=x_eye_at_dt,
        eye_at_dt_sequence=x_eye_at_dt_seq,
        eye_evidence=x_eye_ev,
        eeg=x_eeg,
        eeg_graph=np.concatenate([x_eeg, x_graph], axis=1),
        fusion=np.concatenate([x_eye_at_dt, x_eeg], axis=1),
        fusion_graph=np.concatenate([x_eye_at_dt, x_eye_ev, x_eeg, x_graph], axis=1),
    )


def subset_feature_store(features: FeatureStore, indices: np.ndarray) -> FeatureStore:
    return FeatureStore(**{field.name: getattr(features, field.name)[indices] for field in fields(FeatureStore)})


def make_context(features: FeatureStore, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> ExperimentContext:
    return ExperimentContext(
        features=features,
        train_idx=train_idx,
        test_idx=test_idx,
        y_train=y[train_idx],
        y_test=y[test_idx],
    )


def run_context(
    ctx: ExperimentContext,
    horizon: int,
    protocol: str,
    subject: str,
    seed: int,
    quick: bool,
    method_names: Sequence[str] | None,
) -> List[dict]:
    print(
        f"[INFO] protocol={protocol} horizon={horizon}ms subject={subject} "
        f"train={len(ctx.train_idx)} test={len(ctx.test_idx)} "
        f"test_labels={np.bincount(ctx.y_test, minlength=3).tolist()}"
    )

    rows = []
    for method_impl in build_selected_methods(seed, quick, method_names):
        print(f"[RUN] {protocol} {horizon}ms {subject} {method_impl.name}")
        pred = method_impl.predict(ctx)
        acc, macro_f1 = evaluate_predictions(ctx.y_test, pred)
        print(
            f"[OK] {protocol} {horizon}ms {subject} {method_impl.name}: "
            f"Acc={acc:.2f} Macro-F1={macro_f1:.2f}"
        )
        rows.append(
            {
                "Protocol": protocol,
                "Subject": subject,
                "Type": method_impl.method_type,
                "Method": method_impl.name,
                "Year": method_impl.year,
                "Horizon": horizon,
                "TrainSize": len(ctx.train_idx),
                "TestSize": len(ctx.test_idx),
                "Acc": acc,
                "Macro-F1": macro_f1,
            }
        )
    return rows


def run_pooled_horizon(
    data_dir: Path,
    horizon: int,
    seed: int,
    test_size: float,
    quick: bool,
    method_names: Sequence[str] | None,
    subjects: Sequence[str] | None,
    tasks: Sequence[int] | None,
) -> List[dict]:
    dataset = load_horizon(data_dir, horizon, subjects=subjects, tasks=tasks)
    subject_list = sorted(np.unique(dataset.subjects).tolist())
    task_list = sorted(np.unique(dataset.tasks).astype(int).tolist())
    print(
        f"[INFO] protocol=pooled horizon={horizon}ms samples={len(dataset.y)} "
        f"subjects={subject_list} tasks={task_list}"
    )

    train_idx, test_idx = make_split(dataset.y, dataset.groups, seed, test_size)
    features = build_feature_store(dataset)
    ctx = make_context(features, dataset.y, train_idx, test_idx)
    rows = run_context(
        ctx=ctx,
        horizon=horizon,
        protocol="pooled",
        subject="ALL",
        seed=seed,
        quick=quick,
        method_names=method_names,
    )
    for row in rows:
        row.pop("Protocol", None)
        row.pop("Subject", None)
        row.pop("TrainSize", None)
        row.pop("TestSize", None)
    del dataset, features, ctx
    gc.collect()
    return rows


def run_subject_protocol_horizon(
    data_dir: Path,
    horizon: int,
    protocol: str,
    seed: int,
    test_size: float,
    quick: bool,
    method_names: Sequence[str] | None,
    subjects: Sequence[str] | None,
    tasks: Sequence[int] | None,
) -> List[dict]:
    dataset = load_horizon(data_dir, horizon, subjects=subjects, tasks=tasks)
    subject_list = sorted(np.unique(dataset.subjects).tolist())
    task_list = sorted(np.unique(dataset.tasks).astype(int).tolist())
    if protocol == "cross_subject" and len(subject_list) < 2:
        raise ValueError("Cross-subject evaluation requires at least two subjects.")
    print(
        f"[INFO] protocol={protocol} horizon={horizon}ms "
        f"samples={len(dataset.y)} subjects={subject_list} tasks={task_list}"
    )

    features = build_feature_store(dataset)
    rows = []
    for subject in subject_list:
        subject_mask = dataset.subjects == subject
        if protocol == "within_subject":
            subject_indices = np.flatnonzero(subject_mask)
            subject_features = subset_feature_store(features, subject_indices)
            subject_y = dataset.y[subject_indices]
            subject_groups = dataset.groups[subject_indices]
            train_idx, test_idx = make_split(subject_y, subject_groups, seed, test_size)
            ctx = make_context(subject_features, subject_y, train_idx, test_idx)
        else:
            train_idx = np.flatnonzero(~subject_mask)
            test_idx = np.flatnonzero(subject_mask)
            ctx = make_context(features, dataset.y, train_idx, test_idx)

        rows.extend(
            run_context(
                ctx=ctx,
                horizon=horizon,
                protocol=protocol,
                subject=subject,
                seed=seed,
                quick=quick,
                method_names=method_names,
            )
        )
        del ctx
        if protocol == "within_subject":
            del subject_features
        gc.collect()

    del dataset, features
    gc.collect()
    return rows


def run_experiment(
    data_dir: Path,
    out_dir: Path,
    seed: int,
    test_size: float,
    quick: bool,
    protocol: str,
    method_names: Sequence[str] | None,
    subjects: Sequence[str] | None,
    tasks: Sequence[int] | None,
) -> None:
    all_rows: List[dict] = []
    protocols = requested_protocols(protocol)
    for protocol_name in protocols:
        for horizon in [500, 250, 0]:
            if protocol_name == "pooled":
                all_rows.extend(
                    run_pooled_horizon(
                        data_dir=data_dir,
                        horizon=horizon,
                        seed=seed,
                        test_size=test_size,
                        quick=quick,
                        method_names=method_names,
                        subjects=subjects,
                        tasks=tasks,
                    )
                )
            else:
                all_rows.extend(
                    run_subject_protocol_horizon(
                        data_dir=data_dir,
                        horizon=horizon,
                        protocol=protocol_name,
                        seed=seed,
                        test_size=test_size,
                        quick=quick,
                        method_names=method_names,
                        subjects=subjects,
                        tasks=tasks,
                    )
                )
            gc.collect()
    write_outputs(all_rows, out_dir, selected_method_rows(method_names))


def main() -> None:
    args = parse_args()
    run_experiment(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        seed=args.seed,
        test_size=args.test_size,
        quick=args.quick,
        protocol=args.protocol,
        method_names=args.methods,
        subjects=args.subjects,
        tasks=args.tasks,
    )


if __name__ == "__main__":
    main()
