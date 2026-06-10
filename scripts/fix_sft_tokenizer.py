#!/usr/bin/env python3
"""
fix_sft_tokenizer.py — Repair the SFT checkpoint's tokenizer_config.json IN PLACE.
(Written by Claude Opus 4.8, 2026-06-08.)

The SFT checkpoint's OPERATIVE tokenizer (tokenizer.json) is correct — it has all 26
special tokens incl <think>/</think>/<|im_end|>. Only tokenizer_config.json metadata is
malformed and trips newer transformers / vLLM:
  - extra_special_tokens = a 13-item LIST  (must be a dict; tf4.56 crashes '.keys()')
  - added_tokens_decoder = EMPTY           (should mirror tokenizer.json's 26 added_tokens)
  - chat_template missing from the config   (the jinja lives in chat_template.jinja)

This rebuilds added_tokens_decoder FROM tokenizer.json (the source of truth), drops the
bad extra_special_tokens, and embeds the chat template. Does NOT touch tokenizer.json.

Usage: python fix_sft_tokenizer.py /path/to/model_dir
"""
import json, sys
from pathlib import Path

d = Path(sys.argv[1])
cfg_p = d / "tokenizer_config.json"
tj_p = d / "tokenizer.json"
jinja_p = d / "chat_template.jinja"

cfg = json.load(open(cfg_p))
tj = json.load(open(tj_p))

# 1. Rebuild added_tokens_decoder from tokenizer.json's added_tokens (id -> entry).
atd = {}
for t in tj.get("added_tokens", []):
    atd[str(t["id"])] = {
        "content": t["content"],
        "lstrip": t.get("lstrip", False),
        "normalized": t.get("normalized", False),
        "rstrip": t.get("rstrip", False),
        "single_word": t.get("single_word", False),
        "special": t.get("special", True),
    }
before_atd = len(cfg.get("added_tokens_decoder", {}) or {})
cfg["added_tokens_decoder"] = atd

# 2. Drop the malformed extra_special_tokens list.
had_est = cfg.pop("extra_special_tokens", None)

# 3. Embed chat template if present as a sidecar jinja and absent from config.
embedded_template = False
if jinja_p.exists() and not cfg.get("chat_template"):
    cfg["chat_template"] = jinja_p.read_text()
    embedded_template = True

json.dump(cfg, open(cfg_p, "w"), ensure_ascii=False, indent=2)
print(f"added_tokens_decoder: {before_atd} -> {len(atd)}")
print(f"extra_special_tokens dropped: {type(had_est).__name__} (len={len(had_est) if isinstance(had_est,(list,dict)) else had_est})")
print(f"chat_template embedded from jinja: {embedded_template}; present now: {bool(cfg.get('chat_template'))}")
print(f"eos_token kept: {cfg.get('eos_token')}")
