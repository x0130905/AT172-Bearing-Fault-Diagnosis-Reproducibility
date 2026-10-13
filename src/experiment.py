"""Exact AT172 NumPy experiment with explicit paths and non-destructive outputs."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat

FILES = {
    "normal": [97, 98, 99, 100],
    "ir007": [105, 106, 107, 108],
    "ball007": [118, 119, 120, 121],
    "or007": [130, 131, 132, 133],
}
CLASSES = list(FILES)
SEEDS = [20260916, 20260917, 20260918, 42, 123]
METHODS = ["baseline", "coral", "mixup", "mmd"]
CFG = {
    "sampling_frequency_hz": 12000,
    "window_length": 1024,
    "stride": 1024,
    "windows_per_recording": 80,
    "source_train_loads_hp": [0, 1],
    "validation_load_hp": 2,
    "unseen_test_load_hp": 3,
    "epochs": 30,
    "batch_per_domain": 32,
    "learning_rate": 0.002,
    "conv_channels": 16,
    "kernel_size": 9,
    "coral_lambda": 1.0,
    "mixup_alpha": 0.2,
    "mmd_lambda": 1.0,
    "normalization": "per-window z-score using only that window",
}


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def de_signal(path: Path) -> np.ndarray:
    file_id = int(path.stem)
    wanted = [f"X{file_id:03d}_DE_time", f"X{file_id}_DE_time"]
    obj = loadmat(str(path), variable_names=wanted)
    keys = [key for key in obj if key.endswith("DE_time")]
    if len(keys) != 1:
        raise RuntimeError(f"Expected exactly one DE_time vector in {path}; got {keys}")
    return np.asarray(obj[keys[0]], dtype=np.float64).reshape(-1)


def load_dataset(data_dir: Path):
    xs = {load: [] for load in range(4)}
    ys = {load: [] for load in range(4)}
    rows = []
    length = CFG["window_length"]
    stride = CFG["stride"]
    limit = CFG["windows_per_recording"]
    for class_id, (label, ids) in enumerate(FILES.items()):
        for load, file_id in enumerate(ids):
            path = data_dir / f"{file_id}.mat"
            if not path.exists():
                raise FileNotFoundError(path)
            signal = de_signal(path)
            starts = np.arange(0, len(signal) - length + 1, stride)[:limit]
            if len(starts) != limit:
                raise RuntimeError(f"{path.name} has only {len(starts)} complete windows")
            split = "train" if load in (0, 1) else ("validation" if load == 2 else "unseen_test")
            for window_index, start in enumerate(starts):
                start = int(start)
                xs[load].append(signal[start:start + length, None])
                ys[load].append(class_id)
                rows.append({
                    "load_hp": load, "file_id": file_id, "class": label,
                    "class_id": class_id, "window_index": window_index,
                    "start": start, "end_exclusive": start + length, "split": split,
                })
    for load in range(4):
        x = np.asarray(xs[load], dtype=np.float64)
        y = np.asarray(ys[load], dtype=np.int64)
        mu = x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1, keepdims=True)
        if not np.all(sd > 0):
            raise RuntimeError(f"Zero-variance window in load {load}")
        xs[load] = (x - mu) / sd
        ys[load] = y
    return xs, ys, rows


def softmax(logits):
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


class CNN:
    def __init__(self, seed: int):
        rng = np.random.default_rng(seed)
        k, channels = CFG["kernel_size"], CFG["conv_channels"]
        self.p = {
            "cw": rng.normal(0, np.sqrt(2 / k), (k, 1, channels)),
            "cb": np.zeros(channels),
            "dw": rng.normal(0, np.sqrt(2 / channels), (channels, len(CLASSES))),
            "db": np.zeros(len(CLASSES)),
        }

    @property
    def parameter_count(self):
        return int(sum(value.size for value in self.p.values()))

    def forward(self, x):
        windows = np.lib.stride_tricks.sliding_window_view(x, CFG["kernel_size"], axis=1).transpose(0, 1, 3, 2)
        pre = np.einsum("nlkc,kco->nlo", windows, self.p["cw"], optimize=True) + self.p["cb"]
        features = np.maximum(pre, 0).mean(axis=1)
        return features @ self.p["dw"] + self.p["db"], (windows, pre, features)

    def hard_gradients(self, x, y, domains, coral_weight=0.0, mmd_weight=0.0):
        logits, (windows, pre, features) = self.forward(x)
        probability = softmax(logits)
        dz = probability.copy()
        dz[np.arange(len(y)), y] -= 1
        dz /= len(y)
        dfeatures = dz @ self.p["dw"].T
        ce = float(-np.log(np.maximum(probability[np.arange(len(y)), y], 1e-300)).mean())
        regularizer = 0.0
        groups = [np.flatnonzero(domains == domain) for domain in (0, 1)]
        if coral_weight:
            centered, covariances = [], []
            for idx in groups:
                centered_feature = features[idx] - features[idx].mean(axis=0, keepdims=True)
                centered.append(centered_feature)
                covariances.append(centered_feature.T @ centered_feature / (len(idx) - 1))
            difference = covariances[0] - covariances[1]
            dimension = features.shape[1]
            regularizer = float(np.sum(difference ** 2) / (4 * dimension ** 2))
            for sign, idx, centered_feature in ((1, groups[0], centered[0]), (-1, groups[1], centered[1])):
                dfeatures[idx] += coral_weight * sign * (centered_feature @ difference) / (dimension ** 2 * (len(idx) - 1))
        elif mmd_weight:
            difference = features[groups[0]].mean(axis=0) - features[groups[1]].mean(axis=0)
            dimension = features.shape[1]
            regularizer = float(np.sum(difference ** 2) / dimension)
            dfeatures[groups[0]] += mmd_weight * (2.0 / (dimension * len(groups[0]))) * difference
            dfeatures[groups[1]] -= mmd_weight * (2.0 / (dimension * len(groups[1]))) * difference
        dpre = dfeatures[:, None, :] / pre.shape[1] * (pre > 0)
        gradients = {
            "cw": np.einsum("nlkc,nlo->kco", windows, dpre, optimize=True),
            "cb": dpre.sum(axis=(0, 1)),
            "dw": features.T @ dz,
            "db": dz.sum(axis=0),
        }
        weight = coral_weight or mmd_weight
        return ce + weight * regularizer, ce, regularizer, gradients

    def soft_gradients(self, x, targets):
        logits, (windows, pre, features) = self.forward(x)
        probability = softmax(logits)
        dz = (probability - targets) / len(targets)
        dfeatures = dz @ self.p["dw"].T
        ce = float(-(targets * np.log(np.maximum(probability, 1e-300))).sum(axis=1).mean())
        dpre = dfeatures[:, None, :] / pre.shape[1] * (pre > 0)
        gradients = {
            "cw": np.einsum("nlkc,nlo->kco", windows, dpre, optimize=True),
            "cb": dpre.sum(axis=(0, 1)),
            "dw": features.T @ dz,
            "db": dz.sum(axis=0),
        }
        return ce, ce, 0.0, gradients


def scores(model: CNN, x, y):
    probability = softmax(model.forward(x)[0])
    prediction = probability.argmax(axis=1)
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for actual, predicted in zip(y, prediction):
        cm[actual, predicted] += 1
    precision, recall, f1 = [], [], []
    for class_id in range(len(CLASSES)):
        tp = cm[class_id, class_id]
        pr = tp / max(1, cm[:, class_id].sum())
        rc = tp / max(1, cm[class_id].sum())
        precision.append(pr)
        recall.append(rc)
        f1.append(2 * pr * rc / max(1e-15, pr + rc))
    metrics = {
        "accuracy": float(np.mean(prediction == y)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "confusion_matrix": cm.tolist(),
    }
    return metrics, probability, prediction


def train(seed: int, xs, ys, method: str):
    model = CNN(seed)
    # This reproduces the retained experiment exactly. Mixup consumes extra draws
    # from this same generator, so later source permutations are method-specific.
    rng = np.random.default_rng(seed)
    first_epoch_orders = None
    moment1 = {key: np.zeros_like(value) for key, value in model.p.items()}
    moment2 = {key: np.zeros_like(value) for key, value in model.p.items()}
    step, history, best, best_val, patience = 0, [], None, -1.0, 0
    for epoch in range(CFG["epochs"]):
        orders = {domain: rng.permutation(len(ys[domain])) for domain in (0, 1)}
        if first_epoch_orders is None:
            first_epoch_orders = {domain: order.copy() for domain, order in orders.items()}
        n = min(map(len, orders.values()))
        for start in range(0, n, CFG["batch_per_domain"]):
            ids = {domain: orders[domain][start:start + CFG["batch_per_domain"]] for domain in (0, 1)}
            if min(map(len, ids.values())) < 2:
                continue
            x = np.concatenate([xs[domain][ids[domain]] for domain in (0, 1)])
            y = np.concatenate([ys[domain][ids[domain]] for domain in (0, 1)])
            domains = np.concatenate([np.full(len(ids[domain]), domain) for domain in (0, 1)])
            if method == "mixup":
                lam = float(rng.beta(CFG["mixup_alpha"], CFG["mixup_alpha"]))
                permutation = rng.permutation(len(y))
                onehot = np.eye(len(CLASSES))[y]
                result = model.soft_gradients(
                    lam * x + (1 - lam) * x[permutation],
                    lam * onehot + (1 - lam) * onehot[permutation],
                )
            elif method == "coral":
                result = model.hard_gradients(x, y, domains, coral_weight=CFG["coral_lambda"])
            elif method == "mmd":
                result = model.hard_gradients(x, y, domains, mmd_weight=CFG["mmd_lambda"])
            else:
                result = model.hard_gradients(x, y, domains)
            loss, ce, regularizer, gradients = result
            step += 1
            for key in gradients:
                moment1[key] = .9 * moment1[key] + .1 * gradients[key]
                moment2[key] = .999 * moment2[key] + .001 * gradients[key] ** 2
                corrected1 = moment1[key] / (1 - .9 ** step)
                corrected2 = moment2[key] / (1 - .999 ** step)
                model.p[key] -= CFG["learning_rate"] * corrected1 / (np.sqrt(corrected2) + 1e-8)
        validation_f1 = scores(model, xs[2], ys[2])[0]["macro_f1"]
        history.append({
            "epoch": epoch + 1, "loss": loss, "cross_entropy": ce,
            "regularizer": regularizer, "validation_macro_f1": validation_f1,
        })
        if validation_f1 > best_val + 1e-12:
            best_val = validation_f1
            best = {key: value.copy() for key, value in model.p.items()}
            patience = 0
        else:
            patience += 1
        if patience >= 8:
            break
    model.p = best
    return model, history, first_epoch_orders


def run(data_dir: Path, output_dir: Path, methods=METHODS, seeds=SEEDS):
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    xs, ys, window_rows = load_dataset(data_dir)
    with (output_dir / "window_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(window_rows[0]))
        writer.writeheader(); writer.writerows(window_rows)
    all_runs = []
    for method in methods:
        for seed in seeds:
            print(f"TRAIN {method} {seed}", flush=True)
            model, history, _ = train(seed, xs, ys, method)
            result = {"method": method, "seed": seed}
            for load, name in ((0, "source_0"), (1, "source_1"), (2, "validation_2")):
                result[name] = scores(model, xs[load], ys[load])[0]
            test_metrics, probability, prediction = scores(model, xs[3], ys[3])
            result["unseen_test_3"] = test_metrics
            run_dir = output_dir / f"{method}_{seed}"
            run_dir.mkdir()
            np.savez(run_dir / "checkpoint.npz", **model.p)
            np.savez(run_dir / "predictions_test.npz", y=ys[3], probability=probability, prediction=prediction)
            dump(run_dir / "history.json", history)
            dump(run_dir / "metrics.json", result)
            all_runs.append(result)
    dump(output_dir / "all_runs.json", all_runs)
    dump(output_dir / "run_metadata.json", {
        "elapsed_seconds": time.perf_counter() - started,
        "parameter_count": CNN(0).parameter_count,
        "test_used_for_training_or_selection": False,
        "methods": list(methods), "seeds": list(seeds), "config": CFG,
        "historical_environment_claim": "UNKNOWN",
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()
    run(args.data_dir.resolve(), args.output_dir.resolve(), args.methods, args.seeds)


if __name__ == "__main__":
    main()
