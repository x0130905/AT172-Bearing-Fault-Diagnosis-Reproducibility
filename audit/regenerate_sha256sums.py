"""Generate an LF-stable checksum manifest for the repository release."""

from __future__ import annotations

import hashlib
from pathlib import Path


root = Path(__file__).resolve().parents[1]
destination = root / "SHA256SUMS.txt"
excluded_parts = {".git", "__pycache__"}
lines = []
for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
    if not path.is_file() or path == destination:
        continue
    relative = path.relative_to(root)
    if any(part in excluded_parts for part in relative.parts) or path.suffix.lower() == ".pyc":
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    lines.append(f"{digest}  {relative.as_posix()}")
destination.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
print(f"wrote {len(lines)} entries to {destination}")
