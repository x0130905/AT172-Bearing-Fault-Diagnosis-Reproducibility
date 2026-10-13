"""Capture the controlled verification environment without inferring history."""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
from PIL import __version__ as pillow_version


def sanitize(value):
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if "directory" in lowered or lowered in {"path", "commands"}:
                cleaned[key] = "<redacted-build-path-or-command>"
            else:
                cleaned[key] = sanitize(item)
        return cleaned
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    return value


interesting = [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYTHONHASHSEED",
]

config = getattr(np.__config__, "CONFIG", {})
report = {
    "captured_at_local": "2026-10-14 Asia/Shanghai",
    "scope": "controlled 2026-10-14 verification and fresh-rerun environment; not the historical training environment",
    "historical_environment": {
        "python": "UNKNOWN",
        "numpy": "UNKNOWN",
        "scipy": "UNKNOWN",
        "blas": "UNKNOWN",
        "cpu_thread_settings": "UNKNOWN",
        "reason": "No contemporaneous version lock, interpreter capture, BLAS record, or thread-variable log was found in the original experiment artifacts.",
    },
    "verification_environment": {
        "operating_system": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "UNKNOWN",
        "logical_cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": "<verification-venv>/python.exe",
        "packages": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pillow": pillow_version,
        },
        "numpy_configuration": sanitize(config),
        "thread_and_hash_environment": {name: os.environ.get(name, "UNSET") for name in interesting},
        "random_generator": "numpy.random.default_rng (PCG64 in this NumPy version)",
        "gpu": "Not used; CPU NumPy implementation",
        "floating_point": "float64",
    },
    "declared_seeds": [20260916, 20260917, 20260918, 42, 123],
}

destination = Path(__file__).resolve().parent / "environment_verification_20261014.json"
destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(destination)
