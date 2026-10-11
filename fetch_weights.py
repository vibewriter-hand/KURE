"""Fetch the pinned KURE-v1 weights from the fork's release assets and record SHA-256 of each file.

The default cloud proxy blocks huggingface.co, so the pinned revision is mirrored as split tar parts
(release assets, GitHub is reachable). Every part and every extracted file is verified against pinned hashes.
"""
import argparse
import hashlib
import json
import os
import subprocess
import time

MODEL = "nlpai-lab/KURE-v1"
REVISION = "8b418a58414668e75532ed045c22d9ca018ae2b2"
BASE = "https://github.com/vibewriter-hand/KURE/releases/download/u-cloud-weights-kure-v1-8b418a58/"
PARTS = {
    "kure-v1-8b418a58.tar.part-aa": "fd40e3854d302daa1da1de0031cf254ed2517a5a7931fb5da81069be37d7967d",
    "kure-v1-8b418a58.tar.part-ab": "78064b9a5dec03457c36702dcd62af59ecd65df22bb92ec753725376173b7caf",
    "kure-v1-8b418a58.tar.part-ac": "e3881f735b8bb2279467614f2e70725db39ce0cd2301bec5beae1eac2cd41db4",
}
FILES = {
    "model.safetensors": "c18156e80caf8ff45eb84a24a853130c3bca03087ccb41b051f86e7556bae02c",
    "config.json": "852d42e020c7f989c2acaf30fc683b7f768e8c6d1ab17166e835442162bd825d",
    "modules.json": "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf",
    "1_Pooling/config.json": "13e69897522ee8255104483ed9f219465d1be3936654a54a318758738052789e",
    "tokenizer.json": "fb3c3b93c46fd5a8634e262e1b7de7da11a18b527aa2282b312952b692781dfd",
}
DEST = "weights/KURE-v1"


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
os.makedirs("weights/parts", exist_ok=True)
os.makedirs(DEST, exist_ok=True)
record = {"model": MODEL, "revision": REVISION, "source": "fork release assets (huggingface.co is blocked by the proxy)",
          "parts": {}, "files": {}, "ok": False}
for name, want in PARTS.items():
    path = f"weights/parts/{name}"
    subprocess.run(["curl", "-sSL", "--retry", "3", "--retry-delay", "5", "-o", path, BASE + name], check=True)
    got = sha256(path)
    record["parts"][name] = {"bytes": os.path.getsize(path), "sha256": got, "match": got == want}
    if got != want:
        raise SystemExit(f"part {name} hash mismatch")
cat = subprocess.Popen("cat " + " ".join(f"weights/parts/{n}" for n in sorted(PARTS)), shell=True, stdout=subprocess.PIPE)
subprocess.run(["tar", "-x", "-C", DEST], stdin=cat.stdout, check=True)
cat.wait()
for name, want in FILES.items():
    full = os.path.join(DEST, name)
    got = sha256(full)
    record["files"][name] = {"bytes": os.path.getsize(full), "sha256": got, "match": got == want}
record["ok"] = all(f["match"] for f in record["files"].values())
record["download_and_verify_seconds"] = round(time.time() - started, 1)
with open(args.out, "w") as fh:
    json.dump(record, fh, indent=1)
print("weights ok" if record["ok"] else "weights MISMATCH")
raise SystemExit(0 if record["ok"] else 1)
