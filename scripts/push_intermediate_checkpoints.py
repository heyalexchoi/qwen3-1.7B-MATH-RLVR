#!/usr/bin/env python3
"""
Push intermediate SFT checkpoints (500, 1000) to a private HF backup repo.
Repo: heyalexchoi/qwen3-1.7b-math-sft-checkpoints (private)
"""
import os
import sys
from huggingface_hub import HfApi, create_repo

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set", flush=True)
    sys.exit(1)

REPO_ID = "heyalexchoi/qwen3-1.7b-math-sft-checkpoints"
CHECKPOINTS_DIR = "/home/dev/.openclaw/workspace/qwen3-math-rlvr/outputs/sft_checkpoint"
CHECKPOINTS = ["checkpoint-500", "checkpoint-1000"]

api = HfApi(token=HF_TOKEN)

# Create private repo (no-op if already exists)
print(f"Creating private repo: {REPO_ID}", flush=True)
create_repo(REPO_ID, token=HF_TOKEN, repo_type="model", private=True, exist_ok=True)
print("Repo ready.", flush=True)

for ckpt in CHECKPOINTS:
    local_path = os.path.join(CHECKPOINTS_DIR, ckpt)
    print(f"\n--- Uploading {ckpt} ---", flush=True)
    print(f"Local path: {local_path}", flush=True)

    # List files to upload
    files = []
    for root, dirs, filenames in os.walk(local_path):
        for fname in filenames:
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, local_path)
            size_mb = os.path.getsize(fpath) / (1024**2)
            files.append((fpath, rel, size_mb))
            print(f"  {rel}: {size_mb:.1f} MB", flush=True)

    total_mb = sum(s for _, _, s in files)
    print(f"Total: {total_mb:.1f} MB", flush=True)

    api.upload_folder(
        folder_path=local_path,
        repo_id=REPO_ID,
        repo_type="model",
        path_in_repo=ckpt,
        token=HF_TOKEN,
        commit_message=f"Add {ckpt}",
    )
    print(f"✅ {ckpt} uploaded successfully.", flush=True)

print("\n✅ All intermediate checkpoints pushed to HF.", flush=True)
print(f"Repo: https://huggingface.co/{REPO_ID}", flush=True)
