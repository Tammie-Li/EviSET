"""Metrics, probabilistic fusion helpers, and CSV result rendering."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    acc = accuracy_score(y_true, y_pred) * 100.0
    macro_f1 = f1_score(y_true, y_pred, average="macro") * 100.0
    return acc, macro_f1


def entropy_weighted_average(prob_a: np.ndarray, prob_b: np.ndarray, power: float = 1.0) -> np.ndarray:
    def reliability(p: np.ndarray) -> np.ndarray:
        entropy = -(p * np.log(np.clip(p, 1e-8, 1.0))).sum(axis=1) / np.log(p.shape[1])
        return np.power(1.0 - entropy + 1e-6, power)

    r_a = reliability(prob_a)
    r_b = reliability(prob_b)
    denom = r_a + r_b + 1e-8
    return (prob_a * r_a[:, None] + prob_b * r_b[:, None]) / denom[:, None]


def evidence_probs_from_eye(evidence: np.ndarray) -> np.ndarray:
    e = evidence[:, :3]
    s = e.sum(axis=1, keepdims=True) + 3.0
    belief = e / s
    uncertainty = 3.0 / s
    return belief + uncertainty / 3.0


def summarize_protocol(df: pd.DataFrame, method_rows: Sequence[Tuple[str, str, str]]) -> pd.DataFrame:
    summary_rows = []
    for typ, method, year in method_rows:
        for horizon in [500, 250, 0]:
            hit = df[(df["Method"] == method) & (df["Horizon"] == horizon)]
            if hit.empty:
                continue
            acc = hit["Acc"].astype(float)
            macro_f1 = hit["Macro-F1"].astype(float)
            summary_rows.append(
                {
                    "Protocol": hit["Protocol"].iloc[0],
                    "Type": typ,
                    "Method": method,
                    "Year": year,
                    "Horizon": horizon,
                    "N": int(hit["Subject"].nunique()),
                    "Acc_mean": float(acc.mean()),
                    "Acc_std": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
                    "Macro-F1_mean": float(macro_f1.mean()),
                    "Macro-F1_std": float(macro_f1.std(ddof=1)) if len(macro_f1) > 1 else 0.0,
                }
            )
    return pd.DataFrame(summary_rows)


def _write_pooled_summary(df: pd.DataFrame, out_dir: Path, method_rows: Sequence[Tuple[str, str, str]]) -> None:
    rows = []
    for typ, method, year in method_rows:
        row = {"Type": typ, "Method": method, "Year": year}
        for horizon in [500, 250, 0]:
            hit = df[(df["Method"] == method) & (df["Horizon"] == horizon)].iloc[0]
            row[f"Acc_{horizon}"] = float(hit["Acc"])
            row[f"Macro-F1_{horizon}"] = float(hit["Macro-F1"])
        rows.append(row)

    summary_path = out_dir / "pooled_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"[DONE] wrote {summary_path}")


def write_outputs(rows: List[dict], out_dir: Path, method_rows: Sequence[Tuple[str, str, str]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = out_dir / "table_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"[DONE] wrote {csv_path}")

    if "Protocol" not in df.columns or "Subject" not in df.columns:
        _write_pooled_summary(df, out_dir, method_rows)
        return

    protocol_meta = {
        "within_subject": (
            "within_subject_summary.csv",
        ),
        "cross_subject": (
            "cross_subject_summary.csv",
        ),
    }

    for protocol in ["within_subject", "cross_subject"]:
        protocol_df = df[df["Protocol"] == protocol]
        if protocol_df.empty:
            continue
        (summary_name,) = protocol_meta[protocol]
        summary = summarize_protocol(protocol_df, method_rows)
        summary_path = out_dir / summary_name
        summary.to_csv(summary_path, index=False)
        print(f"[DONE] wrote {summary_path}")
