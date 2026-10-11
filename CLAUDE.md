# Network probe

Task: record which hosts this VM can reach, then push the result. Nothing else. No credentials, no other files.

1. For each URL below run `curl -sS -m 20 -o /dev/null -w '%{http_code}' -I <url>` (print `ERR:<message>` on failure).
2. Write `results/netprobe.json` as `{"<url>": "<status or ERR>"}`; also record `python --version`, `nproc`, and `pip --version`.
3. `git add results && git commit -m "netprobe" && git push` on the branch you were given. Reply `DONE`.

URLs: https://huggingface.co/api/models/nlpai-lab/KURE-v1 · https://huggingface.co/nlpai-lab/KURE-v1/resolve/8b418a58414668e75532ed045c22d9ca018ae2b2/config.json · https://huggingface.co/nlpai-lab/KURE-v1/resolve/8b418a58414668e75532ed045c22d9ca018ae2b2/model.safetensors · https://cdn-lfs.huggingface.co · https://cas-bridge.xethub.hf.co · https://hf-mirror.com · https://github.com/vibewriter-hand/KURE/releases · https://objects.githubusercontent.com · https://release-assets.githubusercontent.com · https://raw.githubusercontent.com/vibewriter-hand/KURE/main/README.md · https://pypi.org/simple/numpy/ · https://files.pythonhosted.org · https://download.pytorch.org/whl/cpu/ · https://modelscope.cn
