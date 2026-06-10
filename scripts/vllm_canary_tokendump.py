#!/usr/bin/env python3
"""
vllm_canary_tokendump.py — vLLM 572 canary + the TERMINATION DISCRIMINATOR.
(Written by Claude Opus 4.8, 2026-06-08.)

Runs number_theory/572 (answer 9) under vLLM (greedy + 8 samples) on the SFT checkpoint
with the repaired tokenizer, and — the point of this run — DUMPS output token_ids and
checks whether the model EVER emits the stop tokens (eos / </think>).

Discriminator (advisor-flagged):
  - emits </think>|eos but generation continued to the cap  -> vLLM didn't honor the stop
    (a config/stop-token issue; a clean tokenizer/eos setup would fix it).
  - NEVER emits </think>|eos                                 -> the MODEL won't terminate
    under vLLM (numerics/precision); a tokenizer fix won't save it -> plan v2 evals on HF.

Usage: python vllm_canary_tokendump.py /path/to/fixed_model_dir
"""
import json, sys
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import vllm as _v

MODEL = sys.argv[1]
PROBLEM = ("Each of the integers $1, 2, 3, \\dots, 9$ is to be placed in the cells of a "
           "$3\\times 3$ grid... ")  # placeholder; real problem text loaded from dataset below

# Load the actual 572 problem from the MATH-500 test split.
from datasets import load_dataset
ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
rec = next(r for r in ds if r["unique_id"].strip("/").endswith("number_theory/572.json")
           or r.get("unique_id", "").endswith("572.json") and "number_theory" in r["unique_id"])
problem = rec["problem"]
print(f"VERS vllm {_v.__version__}")
print(f"PROBLEM {rec['unique_id']}  (answer 9)")

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
eos_id = tok.eos_token_id
think_close_ids = tok.encode("</think>", add_special_tokens=False)
print(f"eos_token={tok.eos_token!r} id={eos_id}   </think> ids={think_close_ids}")

prompt = tok.apply_chat_template(
    [{"role": "user", "content": problem}], tokenize=False, add_generation_prompt=True)

llm = LLM(model=MODEL, trust_remote_code=True, max_model_len=10240, dtype="bfloat16",
          gpu_memory_utilization=0.9)


def analyze(out, tag):
    o = out.outputs[0]
    ids = list(o.token_ids)
    txt = o.text
    has_box = "\\boxed{" in txt
    has_think_close = "</think>" in txt
    emits_eos = eos_id in ids
    emits_think_close = any(tc in ids for tc in think_close_ids if think_close_ids)
    print(f"[{tag}] ntok={len(ids)} finish={o.finish_reason} boxed={has_box} "
          f"text_has_</think>={has_think_close} emits_eos_id={emits_eos} "
          f"emits_</think>_id={emits_think_close}")
    return {"tag": tag, "ntok": len(ids), "finish": o.finish_reason,
            "has_boxed": has_box, "text_has_think_close": has_think_close,
            "emits_eos_id": emits_eos, "emits_think_close_id": emits_think_close,
            "text_tail": txt[-300:]}


# Match the blessed HF chat FORMAT_DEFAULTS: repetition_penalty=1.05 (the param the first
# buggy canary omitted), top_p 0.95, top_k 20. Pass stop_token_ids explicitly for fidelity.
stop_ids = sorted({eos_id, *(think_close_ids or [])})
results = []
g = llm.generate([prompt], SamplingParams(temperature=0.0, repetition_penalty=1.05,
                                          max_tokens=8192, stop_token_ids=[eos_id]))[0]
results.append(analyze(g, "greedy_rep1.05"))

s = llm.generate([prompt] * 8,
                 SamplingParams(temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.05,
                                max_tokens=8192, stop_token_ids=[eos_id]))
for i, out in enumerate(s):
    results.append(analyze(out, f"sample{i}"))

json.dump({"vllm": _v.__version__, "eos_id": eos_id, "think_close_ids": think_close_ids,
           "rep_penalty": 1.05,
           "results": results}, open("/workspace/vllm_canary_tokendump_rep105.json", "w"), indent=2)

# Verdict
any_emits = any(r["emits_eos_id"] or r["emits_think_close_id"] for r in results)
print("\n=== DISCRIMINATOR ===")
if any_emits:
    print("Model DOES emit a stop token in >=1 run -> if it still ran to cap, vLLM isn't "
          "honoring the stop (config-fixable).")
else:
    print("Model NEVER emits eos/</think> under vLLM -> it won't terminate here "
          "(numerics/model, NOT tokenizer) -> standardize v2 evals on HF.")
print("Saved -> /workspace/vllm_canary_tokendump.json")
