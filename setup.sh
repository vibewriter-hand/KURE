#!/usr/bin/env bash
# Install pinned packages and download pinned KURE-v1 weights; write results/weights.json.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
python -m pip install --quiet "torch==2.14.1" --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -3 \
  || python -m pip install --quiet "torch==2.14.1" 2>&1 | tail -3
python -m pip install --quiet "transformers==5.18.0" "sentence-transformers==6.1.0" huggingface_hub 2>&1 | tail -3
python fetch_weights.py --out results/weights.json
