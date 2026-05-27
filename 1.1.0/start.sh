#!/usr/bin/env bash
streamlit run app.py \
  --server.port $PORT \
  --server.address 0.0.0.0 \
  --server.enableCORS false \
  --server.enableXsrfProtection false
