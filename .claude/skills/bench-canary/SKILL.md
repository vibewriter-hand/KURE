---
name: bench-canary
description: Run the 10-question fp32 CPU embedding agreement canary and push results/canary10_cloud.json. Use only for the canary10 job.
---

Run `bash setup.sh`, then `python canary_embed.py --device cpu --out results/canary10_cloud.json`.

Verify before pushing:
- `results/weights.json` lists `model.safetensors` with a SHA-256 and the pinned revision.
- `results/canary10_cloud.json` has `precision` = `fp32`, `device` = `cpu`, 10 queries, 40 passages.
- `results/*.json` hold no token, key or environment value.

On any failure, write the error to `results/failure.txt`, commit it, push, and reply `FAILED`.
