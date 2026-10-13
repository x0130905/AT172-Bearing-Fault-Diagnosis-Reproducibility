"""Single entry point for AT172 artifact verification and fresh reruns."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(command, env=None):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run([str(value) for value in command], cwd=ROOT, env=env, check=True)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download_data(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    with (ROOT / "data_manifests" / "dataset_manifest.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        destination = data_dir / f"{row['file_id']}.mat"
        if destination.exists() and destination.stat().st_size == int(row["bytes"]) and sha256(destination) == row["sha256"]:
            print("VERIFIED", destination.name)
            continue
        temporary = destination.with_suffix(".part")
        print("DOWNLOAD", row["url"])
        with urllib.request.urlopen(row["url"], timeout=300) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, 1024 * 1024)
        if temporary.stat().st_size != int(row["bytes"]) or sha256(temporary) != row["sha256"]:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Downloaded file failed manifest check: {row['file_id']}.mat")
        temporary.replace(destination)


def verify_artifacts(data_dir=None):
    env = os.environ.copy()
    if data_dir:
        env["AT172_DATA_DIR"] = str(Path(data_dir).resolve())
    run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], env=env)
    run([
        sys.executable, "analysis/recompute_metrics.py",
        "--artifact-root", ROOT, "--output-dir", ROOT / "audit" / "recomputed",
    ])


def fresh_rerun(data_dir: Path, output_dir: Path):
    run([
        sys.executable, "src/experiment.py", "--data-dir", data_dir.resolve(),
        "--output-dir", output_dir.resolve(),
    ])


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p_download = sub.add_parser("download-data")
    p_download.add_argument("--data-dir", type=Path, required=True)
    p_verify = sub.add_parser("verify-artifacts")
    p_verify.add_argument("--data-dir", type=Path)
    p_fresh = sub.add_parser("fresh-rerun")
    p_fresh.add_argument("--data-dir", type=Path, required=True)
    p_fresh.add_argument("--output-dir", type=Path, required=True)
    p_all = sub.add_parser("all")
    p_all.add_argument("--data-dir", type=Path, required=True)
    p_all.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "download-data":
        download_data(args.data_dir)
    elif args.command == "verify-artifacts":
        verify_artifacts(args.data_dir)
    elif args.command == "fresh-rerun":
        fresh_rerun(args.data_dir, args.output_dir)
    else:
        download_data(args.data_dir)
        verify_artifacts(args.data_dir)
        fresh_rerun(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()

