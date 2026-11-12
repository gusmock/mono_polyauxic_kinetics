#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -n "${PYTHON:-}" ]]; then
  runtime_python="$PYTHON"
elif [[ -x .venv/bin/python ]]; then
  runtime_python=.venv/bin/python
elif [[ -x venv/bin/python ]]; then
  runtime_python=venv/bin/python
else
  runtime_python=python3
fi
"$runtime_python" -c 'import sys; sys.exit("Python >=3.12 is required; see docs/LEGACY_STREAMLIT_MAINTENANCE.md") if sys.version_info < (3, 12) else None'
exec "$runtime_python" -m streamlit run app.py \
  --server.port "${PORT:-8501}" \
  --server.address "${HOST:-0.0.0.0}"
