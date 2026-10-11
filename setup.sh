#!/usr/bin/env bash
# Install pinned packages (PyPI only: the proxy blocks download.pytorch.org) and fetch pinned KURE-v1 weights.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
python -m pip install --quiet "torch==2.14.1" "transformers==5.18.0" "sentence-transformers==6.1.0" 2>&1 | tail -3
python fetch_weights.py --out results/weights.json
