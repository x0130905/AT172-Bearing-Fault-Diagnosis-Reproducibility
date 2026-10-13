"""Compare retained artifacts, paper values, and the 2026-10-14 fresh rerun."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import numpy as np


METHODS = ["baseline", "coral", "mixup", "mmd"]
LABELS = {"baseline": "ERM", "coral": "CORAL", "mixup": "Mixup", "mmd": "Linear MMD"}
SEEDS = [20260916, 20260917, 20260918, 42, 123]
PAPER = {
    "baseline": (85.24, 20.41),
    "coral": (85.24, 20.41),
    "mixup": (78.58, 20.45),
    "mmd": (85.37, 20.34),
}


def sign_flip_p(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    permuted = [abs(float(np.mean(values * np.asarray(signs)))) for signs in product((-1.0, 1.0), repeat=len(values))]
    return float(sum(value >= observed - 1e-15 for value in permuted) / len(permuted))


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--fresh-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_rows = []
    values = {"retained": {}, "fresh": {}}
    for method in METHODS:
        values["retained"][method] = []
        values["fresh"][method] = []
        for seed in SEEDS:
            run = f"{method}_{seed}"
            old_pred = np.load(args.artifact_root / "predictions" / run / "predictions_test.npz")
            new_pred = np.load(args.fresh_root / run / "predictions_test.npz")
            old_metrics = load_metrics(args.artifact_root / "metrics" / run / "metrics.json")
            new_metrics = load_metrics(args.fresh_root / run / "metrics.json")
            old_checkpoint = np.load(args.artifact_root / "checkpoints" / run / "checkpoint.npz")
            new_checkpoint = np.load(args.fresh_root / run / "checkpoint.npz")
            old_history = load_metrics(args.artifact_root / "histories" / run / "history.json")
            new_history = load_metrics(args.fresh_root / run / "history.json")

            old_f1 = float(old_metrics["unseen_test_3"]["macro_f1"])
            new_f1 = float(new_metrics["unseen_test_3"]["macro_f1"])
            values["retained"][method].append(old_f1)
            values["fresh"][method].append(new_f1)

            probability_delta = float(np.max(np.abs(old_pred["probability"] - new_pred["probability"])))
            checkpoint_delta = max(float(np.max(np.abs(old_checkpoint[key] - new_checkpoint[key]))) for key in old_checkpoint.files)
            history_delta = max(
                abs(float(left[key]) - float(right[key]))
                for left, right in zip(old_history, new_history)
                for key in left
                if key != "epoch"
            )
            row = {
                "method": LABELS[method],
                "seed": seed,
                "labels_exact": bool(np.array_equal(old_pred["y"], new_pred["y"])),
                "predictions_exact": bool(np.array_equal(old_pred["prediction"], new_pred["prediction"])),
                "metrics_json_exact": old_metrics == new_metrics,
                "retained_macro_f1_pct": 100 * old_f1,
                "fresh_macro_f1_pct": 100 * new_f1,
                "macro_f1_delta_percentage_points": 100 * (new_f1 - old_f1),
                "probability_max_abs_difference": probability_delta,
                "checkpoint_max_abs_difference": checkpoint_delta,
                "history_max_abs_difference": history_delta,
            }
            row["functionally_exact_at_1e-12"] = (
                row["labels_exact"]
                and row["predictions_exact"]
                and row["metrics_json_exact"]
                and probability_delta <= 1e-12
                and checkpoint_delta <= 1e-12
                and history_delta <= 1e-12
            )
            run_rows.append(row)

    with (args.output_dir / "fresh_rerun_comparison_by_run.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(run_rows[0]))
        writer.writeheader()
        writer.writerows(run_rows)

    comparison_rows = []
    for source in ("retained", "fresh"):
        erm = np.asarray(values[source]["baseline"], dtype=float)
        for method in METHODS:
            array = np.asarray(values[source][method], dtype=float)
            mean = 100 * float(array.mean())
            sd = 100 * float(array.std(ddof=1))
            paper_mean, paper_sd = PAPER[method]
            p_value = 1.0 if method == "baseline" else sign_flip_p(array - erm)
            comparison_rows.append(
                {
                    "method": LABELS[method],
                    "source": source,
                    "n_seeds": len(array),
                    "mean_macro_f1_pct": mean,
                    "sample_sd_macro_f1_pct": sd,
                    "paper_reported_mean_pct": paper_mean,
                    "paper_reported_sample_sd_pct": paper_sd,
                    "mean_delta_vs_paper_percentage_points": mean - paper_mean,
                    "sd_delta_vs_paper_percentage_points": sd - paper_sd,
                    "exact_two_sided_sign_flip_p_vs_erm": p_value,
                }
            )

    with (args.output_dir / "numerical_comparison_20261014.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)

    summary = {
        "run_count": len(run_rows),
        "all_five_seeds_present_for_all_methods": len(run_rows) == 20,
        "all_labels_exact": all(row["labels_exact"] for row in run_rows),
        "all_predictions_exact": all(row["predictions_exact"] for row in run_rows),
        "all_metrics_json_exact": all(row["metrics_json_exact"] for row in run_rows),
        "all_runs_functionally_exact_at_1e-12": all(row["functionally_exact_at_1e-12"] for row in run_rows),
        "maximum_probability_absolute_difference": max(row["probability_max_abs_difference"] for row in run_rows),
        "maximum_checkpoint_absolute_difference": max(row["checkpoint_max_abs_difference"] for row in run_rows),
        "maximum_history_absolute_difference": max(row["history_max_abs_difference"] for row in run_rows),
        "mmd_vs_erm_p_retained": next(
            row["exact_two_sided_sign_flip_p_vs_erm"]
            for row in comparison_rows
            if row["source"] == "retained" and row["method"] == "Linear MMD"
        ),
        "mmd_vs_erm_p_fresh": next(
            row["exact_two_sided_sign_flip_p_vs_erm"]
            for row in comparison_rows
            if row["source"] == "fresh" and row["method"] == "Linear MMD"
        ),
    }
    (args.output_dir / "fresh_rerun_comparison_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
