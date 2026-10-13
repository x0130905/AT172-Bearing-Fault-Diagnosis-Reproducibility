#!/usr/bin/env sh
set -eu
if [ "$#" -ne 2 ]; then
  echo "Usage: sh run_all.sh DATA_DIR OUTPUT_DIR" >&2
  exit 2
fi
python reproduce.py all --data-dir "$1" --output-dir "$2"

