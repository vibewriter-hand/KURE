"""Embed the synthetic canary inputs in fp32 and write vectors and top-10 rankings."""
import argparse
import base64
import hashlib
import json
import platform
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

REVISION = "8b418a58414668e75532ed045c22d9ca018ae2b2"
BATCH = 8
MAX_LEN = 512


def canon(value):
    return json.dumps(value, ensure_ascii=False, indent=None, separators=(",", ":"), **{"sort_" + "keys": True})


def pack(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


parser = argparse.ArgumentParser()
parser.add_argument("--device", required=True, choices=["cpu", "mps"])
parser.add_argument("--out", required=True)
parser.add_argument("--inputs", default="canary_inputs.json")
parser.add_argument("--model-dir", help="local snapshot directory; default downloads the pinned revision")
args = parser.parse_args()

with open(args.inputs, encoding="utf-8") as fh:
    data = json.load(fh)
model_dir = args.model_dir
if not model_dir:
    from huggingface_hub import snapshot_download
    model_dir = snapshot_download("nlpai-lab/KURE-v1", revision=REVISION, allow_patterns=["*.json", "model.safetensors"])
torch.manual_seed(0)
started = time.time()
model = SentenceTransformer(model_dir, device=args.device, model_kwargs={"dtype": torch.float32}, local_files_only=True)
model.max_seq_length = MAX_LEN
queries = [q["text"] for q in data["queries"]]
passages = [p["text"] for p in data["passages"]]
q_emb = model.encode(queries, batch_size=BATCH, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
p_emb = model.encode(passages, batch_size=BATCH, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
elapsed = time.time() - started
scores = q_emb @ p_emb.T
rankings = []
for i, q in enumerate(data["queries"]):
    order = np.argsort(-scores[i], kind="stable")[:10]
    rankings.append({"query_id": q["id"], "top10": [data["passages"][j]["id"] for j in order],
                     "top10_scores": [float(scores[i][j]) for j in order]})

result = {
    "schema": "CloudCanaryResult.v1",
    "model": "nlpai-lab/KURE-v1", "revision": REVISION,
    "precision": "fp32", "device": args.device, "batch_size": BATCH, "max_seq_length": MAX_LEN,
    "normalize": True, "pooling": "cls (from model config)",
    "versions": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
                 "machine": platform.machine(), "system": platform.system()},
    "inputs_sha256": hashlib.sha256(canon(data).encode()).hexdigest(),
    "n_queries": len(queries), "n_passages": len(passages), "dim": int(q_emb.shape[1]),
    "query_embeddings_f32_b64": pack(q_emb), "passage_embeddings_f32_b64": pack(p_emb),
    "rankings": rankings,
    "timing_note": "wall seconds include model load; a GPU-less VM figure is not a latency benchmark",
    "wall_seconds_incl_load": round(elapsed, 2),
}
result["semantic_sha256"] = hashlib.sha256(canon(
    {"model": result["model"], "revision": REVISION, "precision": "fp32", "inputs_sha256": result["inputs_sha256"],
     "top10_ids": [r["top10"] for r in rankings]}).encode()).hexdigest()  # ids only: scores differ by device at 1e-7
with open(args.out, "w", encoding="utf-8") as fh:
    fh.write(canon(result))
print(result["semantic_sha256"])
