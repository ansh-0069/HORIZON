#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-./data}"
MODEL_PATH="${2:-./pickle/model.pkl}"
OUTPUT_PATH="${3:-./output/predictions.csv}"

mkdir -p "$(dirname "$OUTPUT_PATH")"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_EXECUTABLE="$PYTHON_BIN"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_EXECUTABLE="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_EXECUTABLE="python"
elif command -v py >/dev/null 2>&1; then
  PYTHON_EXECUTABLE="py"
else
  echo "Python 3 was not found. Set PYTHON_BIN or add python3/python to PATH." >&2
  exit 127
fi

"$PYTHON_EXECUTABLE" -m src.predict \
  --data-dir "$DATA_DIR" \
  --model "$MODEL_PATH" \
  --output "$OUTPUT_PATH"
