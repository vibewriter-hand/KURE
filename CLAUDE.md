# Bench canary pack

Task: run the embedding agreement canary and push the result. Nothing else.

Rules:
- Use only the files in this pack. Do not read other files or add dependencies.
- Inputs are synthetic (`canary_inputs.json`). Do not fetch or write any other data.
- Precision is fp32. Device is cpu. Do not change them.
- Report no latency as a benchmark: this VM has no GPU.
- Do not print tokens, keys or environment values.

Steps:
1. `bash setup.sh` (installs pinned packages from PyPI, fetches the pinned weights from this repo's release assets, verifies hashes, writes `results/weights.json`). huggingface.co and download.pytorch.org are blocked by the proxy; do not try them.
2. `python canary_embed.py --device cpu --out results/canary10_cloud.json`.
3. `git add results && git commit -m "canary10 cloud result"` and push the branch you were given.
4. Reply with one line: `DONE` plus the SHA-256 printed by step 2, or `FAILED` plus the failing step and its last error line.

Skill: `.claude/skills/bench-canary/SKILL.md` has the exact fields to verify.
