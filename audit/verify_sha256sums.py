"""Verify the repository checksum manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


root = Path(__file__).resolve().parents[1]
failures = []
entries = 0
for line in (root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
    digest, relative = line.split("  ", 1)
    entries += 1
    path = root / Path(relative)
    if not path.is_file():
        failures.append({"path": relative, "error": "missing"})
        continue
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != digest:
        failures.append({"path": relative, "expected": digest, "actual": actual})

result = {"entries": entries, "failures": failures, "pass": not failures}
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["pass"] else 1)
