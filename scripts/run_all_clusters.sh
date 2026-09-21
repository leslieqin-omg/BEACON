#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [ -x "/opt/miniconda3/bin/python" ]; then
    PYTHON_BIN="/opt/miniconda3/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    PYTHON_BIN="python"
fi

if ! command -v mmseqs >/dev/null 2>&1; then
    for cand in "${PYTHON_BIN%/python*}/mmseqs" "${HOME}/.local/bin/mmseqs" "${HOME}/miniconda3/bin/mmseqs" "/opt/miniconda3/bin/mmseqs" "/opt/conda/bin/mmseqs"; do
        if [ -x "${cand}" ]; then
            export PATH="$(dirname "${cand}"):${PATH}"
            break
        fi
    done
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "=========================================================================="
echo "  BEACON: One-Click Runner for All Antigen Clustering Benchmarks"
echo "  Experiments: ag40 (40%), ag60 (60%), ag80 (80%)"
echo "  Python: ${PYTHON_BIN}"
echo "=========================================================================="

CONFIGS=("configs/ag40.yaml" "configs/ag60.yaml" "configs/ag80.yaml")

for cfg in "${CONFIGS[@]}"; do
    echo ""
    echo "=========================================================================="
    echo "  [START] Running: ${cfg}"
    echo "=========================================================================="
    "${PYTHON_BIN}" -u scripts/train_kfold.py --config "${cfg}" "$@"
    echo "  [DONE] Finished: ${cfg}"
done

echo ""
echo "=========================================================================="
echo "  Generating Benchmark Plots and Tables..."
echo "=========================================================================="
"${PYTHON_BIN}" -u scripts/plot_benchmarks.py

echo ""
echo "=========================================================================="
echo "  [SUCCESS] All 3 clustering experiments and benchmark reports completed!"
echo "  Reports saved in: artifacts/report/"
echo "=========================================================================="
