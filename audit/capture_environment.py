"""Capture the current verification/rerun environment without claiming history."""
from __future__ import annotations

import contextlib
import io
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
from PIL import __version__ as pillow_version


buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    np.show_config()

interesting = [
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "PYTHONHASHSEED",
]
report = {
    "scope": "current audit and fresh-rerun environment; not the historical training environment",
    "historical_environment": "UNKNOWN",
    "captured_at_local": __import__("datetime").datetime.now().astimezone().isoformat(),
    "operating_system": platform.platform(),
    "machine": platform.machine(),
    "processor": platform.processor() or "UNKNOWN",
    "python": sys.version,
    "python_executable": sys.executable,
    "packages": {"numpy": np.__version__, "scipy": scipy.__version__, "pillow": pillow_version},
    "numpy_configuration": buffer.getvalue(),
    "environment_variables": {name: os.environ.get(name, "UNSET") for name in interesting},
    "random_generators": "numpy.random.default_rng with PCG64 unless changed by NumPy",
    "gpu": "Not used; implementation is CPU NumPy",
}
path = Path(__file__).resolve().parent / "environment_report.json"
path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(path)
