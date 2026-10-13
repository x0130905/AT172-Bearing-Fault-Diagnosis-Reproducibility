"""Recompute AT172 metrics and paper tables from retained prediction arrays."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from statistical_tests import paired_summary

METHODS = ["baseline", "coral", "mixup", "mmd"]
LABELS = {"baseline": "ERM", "coral": "CORAL", "mixup": "Mixup", "mmd": "Linear MMD"}
SEEDS = [20260916, 20260917, 20260918, 42, 123]


def metric_from_predictions(path: Path):
    data = np.load(path)
    y, prediction = np.asarray(data["y"], dtype=int), np.asarray(data["prediction"], dtype=int)
    cm = np.zeros((4, 4), dtype=int)
    for actual, predicted in zip(y, prediction):
        cm[actual, predicted] += 1
    precision, recall, f1 = [], [], []
    for class_id in range(4):
        tp = cm[class_id, class_id]
        pr = tp / max(1, cm[:, class_id].sum())
        rc = tp / max(1, cm[class_id].sum())
        precision.append(pr); recall.append(rc)
        f1.append(2 * pr * rc / max(1e-15, pr + rc))
    return {
        "accuracy": float(np.mean(y == prediction)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "class_recall": recall,
        "confusion_matrix": cm.tolist(),
        "n": int(len(y)),
    }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    artifacts, out = args.artifact_root.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    verification, results, class_rows = [], {}, []
    for method in METHODS:
        results[method] = {}
        for seed in SEEDS:
            run = f"{method}_{seed}"
            calculated = metric_from_predictions(artifacts / "predictions" / run / "predictions_test.npz")
            recorded = json.loads((artifacts / "metrics" / run / "metrics.json").read_text(encoding="utf-8"))["unseen_test_3"]
            results[method][seed] = calculated
            for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
                difference = calculated[metric] - float(recorded[metric])
                verification.append({
                    "method": LABELS[method], "seed": seed, "domain": "unseen_test_3",
                    "metric": metric, "paper_artifact_value": recorded[metric],
                    "recomputed_value": calculated[metric], "difference": difference,
                    "status": "VERIFIED" if abs(difference) <= 1e-12 else "INCONSISTENT",
                    "evidence": f"predictions/{run}/predictions_test.npz",
                })
            for class_id, value in enumerate(calculated["class_recall"]):
                class_rows.append({"method": LABELS[method], "seed": seed, "class_id": class_id, "recall": value})
    write_csv(out / "paper_results_verification.csv", verification, list(verification[0]))
    write_csv(out / "class_recall.csv", class_rows, list(class_rows[0]))

    table_iii, table_iv, table_v = [], [], []
    baseline = [results["baseline"][seed]["macro_f1"] for seed in SEEDS]
    for method in METHODS:
        metrics = {}
        for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
            values = np.asarray([results[method][seed][metric] for seed in SEEDS]) * 100
            metrics[metric] = values
        table_iii.append({
            "method": LABELS[method],
            **{f"{metric}_mean_pct": float(values.mean()) for metric, values in metrics.items()},
            **{f"{metric}_sample_sd_pct": float(values.std(ddof=1)) for metric, values in metrics.items()},
        })
        f1 = metrics["macro_f1"]
        if method == "baseline":
            paired = {"mean_difference": 0.0, "exact_two_sided_sign_flip_p": 1.0}
        else:
            paired = paired_summary(f1 / 100, baseline)
        table_iv.append({
            "method": LABELS[method], "median_macro_f1_pct": float(np.median(f1)),
            "minimum_pct": float(f1.min()), "maximum_pct": float(f1.max()),
            "paired_delta_vs_erm_pct": 100 * paired["mean_difference"],
            "exact_two_sided_sign_flip_p": paired["exact_two_sided_sign_flip_p"],
        })
        for seed in SEEDS:
            table_v.append({
                "method": LABELS[method], "seed": seed,
                "accuracy_pct": 100 * results[method][seed]["accuracy"],
                "macro_f1_pct": 100 * results[method][seed]["macro_f1"],
            })

    loss_rows = []
    for method in ("coral", "mmd"):
        histories = []
        for seed in SEEDS:
            histories.extend(json.loads((artifacts / "histories" / f"{method}_{seed}" / "history.json").read_text(encoding="utf-8")))
        ce = np.asarray([row["cross_entropy"] for row in histories], dtype=float)
        reg = np.asarray([row["regularizer"] for row in histories], dtype=float)
        loss_rows.append({
            "objective": LABELS[method], "n": len(histories), "scope": "final source mini-batch logged per recorded epoch",
            "ce_median": np.median(ce), "ce_q1": np.quantile(ce, .25), "ce_q3": np.quantile(ce, .75),
            "penalty_median": np.median(reg), "penalty_q1": np.quantile(reg, .25), "penalty_q3": np.quantile(reg, .75),
            "penalty_ce_median": np.median(reg / ce),
        })
    write_csv(out / "table_ii_loss_scale.csv", loss_rows, list(loss_rows[0]))
    write_csv(out / "table_iii_aggregate.csv", table_iii, list(table_iii[0]))
    write_csv(out / "table_iv_distribution_and_paired.csv", table_iv, list(table_iv[0]))
    write_csv(out / "table_v_all_seed_results.csv", table_v, list(table_v[0]))

    summary = {
        "all_prediction_metrics_match_recorded_metrics": all(row["status"] == "VERIFIED" for row in verification),
        "coral_predictions_equal_erm_all_seeds": all(
            np.array_equal(
                np.load(artifacts / "predictions" / f"coral_{seed}" / "predictions_test.npz")["prediction"],
                np.load(artifacts / "predictions" / f"baseline_{seed}" / "predictions_test.npz")["prediction"],
            ) for seed in SEEDS
        ),
        "mmd_seeds_with_changed_predictions_vs_erm": [
            seed for seed in SEEDS if not np.array_equal(
                np.load(artifacts / "predictions" / f"mmd_{seed}" / "predictions_test.npz")["prediction"],
                np.load(artifacts / "predictions" / f"baseline_{seed}" / "predictions_test.npz")["prediction"],
            )
        ],
        "table_iv": table_iv,
    }
    (out / "verification_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
