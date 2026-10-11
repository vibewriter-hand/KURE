"""Download the pinned model revision and record SHA-256 of each weight file."""
import argparse
import hashlib
import json
import time

from huggingface_hub import snapshot_download

MODEL = "nlpai-lab/KURE-v1"
REVISION = "8b418a58414668e75532ed045c22d9ca018ae2b2"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
args = parser.parse_args()
started = time.time()
path = snapshot_download(MODEL, revision=REVISION, allow_patterns=["*.json", "model.safetensors"])
record = {"model": MODEL, "revision": REVISION, "snapshot_path": path, "download_seconds": round(time.time() - started, 1)}
record["files"] = {}
import os
for name in ("model.safetensors", "config.json", "modules.json", "1_Pooling/config.json", "tokenizer.json"):
    full = os.path.join(path, name)
    record["files"][name] = {"bytes": os.path.getsize(full), "sha256": sha256(full)}
with open(args.out, "w") as fh:
    json.dump(record, fh, indent=1, sort_keys=True)
print(json.dumps(record["files"]["model.safetensors"]))
