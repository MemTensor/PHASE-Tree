#!/usr/bin/env bash
# Minimal venv for GPT judge scoring (openai API only, no GPU stack).
set -euo pipefail

VENV_DIR="${VENV_DIR:-${ROOT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/.venv-judge}"
mkdir -p "$(dirname "$VENV_DIR")"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
    echo "Creating venv at ${VENV_DIR}..."
    PY="${PYTHON_FOR_VENV:-}"
    if [ -z "$PY" ]; then
        for cand in /home/zhlin/miniconda3-py310/bin/python3 python3.10 python3; do
            if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
        done
    fi
    "$PY" -m venv "$VENV_DIR"
fi

echo "Installing judge dependencies..."
"${VENV_DIR}/bin/pip" install -q --upgrade pip
"${VENV_DIR}/bin/pip" install -q 'openai>=1.40.0' 'python-dotenv>=1.0.0' 'tqdm>=4.65'

echo "Verifying..."
"${VENV_DIR}/bin/python" -c "import openai, dotenv, tqdm; print('OK:', openai.__version__)"
