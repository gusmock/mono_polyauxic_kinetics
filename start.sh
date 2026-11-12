#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec "${PYTHON:-python}" -m streamlit run app.py \
  --server.port "${PORT:-8501}" \
  --server.address "${HOST:-0.0.0.0}"
