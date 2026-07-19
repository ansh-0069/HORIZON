#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -gt 3 ]]; then
  echo "Usage: ./run.sh [DATA_DIR] [MODEL_PATH] [OUTPUT_PATH]" >&2
  exit 64
fi

# The guide's local-development contract requires zero-argument execution;
# evaluator-provided arguments still take precedence without any hard-coded
# absolute path or interactive input.
DATA_DIR="${1:-./data}"
MODEL_PATH="${2:-./pickle/model.pkl}"
OUTPUT_PATH="${3:-./output/predictions.csv}"

# Forecast inference evaluates many small linear models.  Allowing a BLAS
# runtime to create a large worker pool for each operation is materially
# slower on shared evaluator machines and can make otherwise deterministic
# execution depend on host scheduling.  One thread is the correct execution
# profile for this small, serial workload.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$(dirname "$OUTPUT_PATH")"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  # PYTHON_BIN is intentionally a single executable path, not a shell
  # command. Quoting it preserves paths containing spaces on Windows/WSL.
  PYTHON_COMMAND=("$PYTHON_BIN")
elif command -v python3 >/dev/null 2>&1 && python3 --version >/dev/null 2>&1; then
  PYTHON_COMMAND=(python3)
elif command -v python >/dev/null 2>&1 && python --version >/dev/null 2>&1; then
  PYTHON_COMMAND=(python)
elif command -v py >/dev/null 2>&1 && py -3 --version >/dev/null 2>&1; then
  # A Windows App Execution Alias can expose `py` even when no interpreter is
  # installed. Probe the requested Python 3 launcher before selecting it.
  PYTHON_COMMAND=(py -3)
else
  echo "Python 3 was not found or is not runnable. Set PYTHON_BIN or add a runnable python3/python to PATH." >&2
  exit 127
fi

"${PYTHON_COMMAND[@]}" -m src.predict \
  --data-dir "$DATA_DIR" \
  --model "$MODEL_PATH" \
  --output "$OUTPUT_PATH"
