#!/usr/bin/env python3
"""
vllm_parity_canary.py — vLLM-vs-HF PARITY experiment (Written by Claude Opus 4.8, 2026-06-10).

The decisive, un-walk-back-able test the earlier canaries lacked: run the SAME problems through
vLLM AND the HF backend on the SAME pod, SAME transformers version, SAME blessed sampling params,
and compare ACCURACY + per-problem AGREEMENT. Every prior vLLM verdict was confounded because the
HF "control" lived on a different pod / tf / params. Here the only variable is the backend.

Blessed params (from scripts/math500_eval.py FORMAT_DEFAULTS['chat']): greedy temperature=0.0,
repetition_penalty=1.05, max_new_tokens=8192, stop on {eos} ∪ {<|im_end|>}.

PRE-COMMITTED DECISION TREE (write it before you see the output):
  * vLLM acc ≈ HF acc AND high per-problem agreement  -> backends agree -> vLLM is faithful and
        usable. LOCK the v2 stack = (this vLLM version + this tf).
  * vLLM degenerates (8192/no-boxed) while HF-on-same-pod is clean -> vLLM-SPECIFIC kernel/numerics
        issue on this checkpoint (NOT tf) -> vLLM not usable for THIS checkpoint; revisit on v2.
  * BOTH degenerate -> it's the fragile v1 checkpoint at this tf, not the backend -> re-test on a
        well-trained v2 before concluding anything about vLLM.

Usage (run BOTH on the pod, then compare):
  python vllm_parity_canary.py --backend hf   --model /workspace/model --n 25
  python vllm_parity_canary.py --backend vllm --model /workspace/model --n 25
  python vllm_parity_canary.py --compare
"""
import argparse, json, sys
from pathlib import Path

OUT_DIR = Path("/workspace")
N_DEFAULT = 25
MAXTOK = 8192
REP = 1.05


def load_problems(n):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = sorted(ds, key=lambda r: r["unique_id"])[:n]
    return [{"id": r["unique_id"], "problem": r["problem"], "expected": r["answer"]} for r in rows]


def mv_ok(pred, exp):
    try:
        from math_verify import parse as mvp, verify as mvv
        if not pred or exp is None:
            return False
        g, a = mvp(f"${exp}$"), mvp(f"${pred}$")
        return bool(g and a and mvv(g, a))
    except Exception:
        return False


def extract_boxed(text):
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return ""
    start = idx + len("\\boxed{")
    depth, out = 1, []
    for ch in text[start:]:
        if ch == "{":
            depth += 1; out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
            out.append(ch)
        else:
            out.append(ch)
    return "".join(out).strip()


def run_vllm(model, probs):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    import vllm, transformers
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    eos_id = tok.eos_token_id
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    stop_ids = sorted({i for i in {eos_id, im_end} if i is not None and i >= 0})
    prompts = [tok.apply_chat_template([{"role": "user", "content": p["problem"]}],
                                       tokenize=False, add_generation_prompt=True) for p in probs]
    llm = LLM(model=model, trust_remote_code=True, max_model_len=10240, dtype="bfloat16",
              gpu_memory_utilization=0.9)
    sp = SamplingParams(temperature=0.0, repetition_penalty=REP, max_tokens=MAXTOK,
                        stop_token_ids=stop_ids)
    outs = llm.generate(prompts, sp)
    recs = []
    for p, o in zip(probs, outs):
        out = o.outputs[0]
        txt = out.text
        boxed = extract_boxed(txt)
        recs.append({"id": p["id"], "expected": p["expected"], "boxed": boxed,
                     "ntok": len(out.token_ids), "finish": out.finish_reason,
                     "degenerate": len(out.token_ids) >= MAXTOK, "tail": txt[-120:]})
    return {"backend": "vllm", "vllm": vllm.__version__, "transformers": transformers.__version__,
            "stop_ids": stop_ids, "records": recs}


def run_hf(model, probs):
    import torch, transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    eos_id = tok.eos_token_id
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    stop_ids = sorted({i for i in {eos_id, im_end} if i is not None and i >= 0})
    m = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True,
                                             torch_dtype=torch.bfloat16, device_map="cuda")
    m.eval()
    prompts = [tok.apply_chat_template([{"role": "user", "content": p["problem"]}],
                                       tokenize=False, add_generation_prompt=True) for p in probs]
    recs = []
    BS = 8
    for i in range(0, len(probs), BS):
        chunk = probs[i:i + BS]
        pchunk = prompts[i:i + BS]
        enc = tok(pchunk, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            gen = m.generate(**enc, max_new_tokens=MAXTOK, do_sample=False,
                             repetition_penalty=REP, eos_token_id=stop_ids,
                             pad_token_id=tok.pad_token_id)
        for j, p in enumerate(chunk):
            new = gen[j][enc["input_ids"].shape[1]:]
            ntok = int((new != tok.pad_token_id).sum())
            txt = tok.decode(new, skip_special_tokens=True)
            boxed = extract_boxed(txt)
            recs.append({"id": p["id"], "expected": p["expected"], "boxed": boxed,
                         "ntok": ntok, "finish": "length" if ntok >= MAXTOK else "stop",
                         "degenerate": ntok >= MAXTOK, "tail": txt[-120:]})
        print(f"  hf {min(i+BS,len(probs))}/{len(probs)}", flush=True)
    return {"backend": "hf", "transformers": transformers.__version__,
            "stop_ids": stop_ids, "records": recs}


def score(blob):
    for r in blob["records"]:
        r["ok"] = mv_ok(r["boxed"], r["expected"])
    n = len(blob["records"])
    acc = sum(r["ok"] for r in blob["records"])
    deg = sum(r["degenerate"] for r in blob["records"])
    blob["acc"] = acc; blob["n"] = n; blob["degenerate_count"] = deg
    return blob


def compare():
    hf = json.load(open(OUT_DIR / "parity_hf.json"))
    vl = json.load(open(OUT_DIR / "parity_vllm.json"))
    hi = {r["id"]: r for r in hf["records"]}
    vi = {r["id"]: r for r in vl["records"]}
    ids = [i for i in hi if i in vi]
    agree = sum(hi[i]["ok"] == vi[i]["ok"] for i in ids)
    box_agree = sum((hi[i]["boxed"] or "∅") == (vi[i]["boxed"] or "∅") for i in ids)
    print("=" * 70)
    print(f"HF   : tf {hf['transformers']}  acc {hf['acc']}/{hf['n']}  degenerate {hf['degenerate_count']}")
    print(f"vLLM : v {vl.get('vllm','?')} tf {vl['transformers']}  acc {vl['acc']}/{vl['n']}  "
          f"degenerate {vl['degenerate_count']}")
    print(f"per-problem correctness agreement: {agree}/{len(ids)}")
    print(f"exact boxed-string agreement:      {box_agree}/{len(ids)}")
    print("=" * 70)
    print(f"{'id':<28}{'exp':<10}{'HF':<14}{'vLLM':<14}{'ok=':<8}")
    for i in ids:
        h, v = hi[i], vi[i]
        flag = "" if h["ok"] == v["ok"] else "  <-- DISAGREE"
        print(f"{i.split('/')[-1]:<28}{str(h['expected'])[:8]:<10}"
              f"{(h['boxed'][:8] or '∅')+('!' if h['degenerate'] else ''):<14}"
              f"{(v['boxed'][:8] or '∅')+('!' if v['degenerate'] else ''):<14}"
              f"{str(h['ok'])[0]}/{str(v['ok'])[0]}{flag}")
    print("\n('!' = hit 8192-token cap / degenerate)")
    # verdict
    print("\n=== VERDICT (against pre-committed tree) ===")
    if vl["acc"] >= max(1, int(0.9 * hf["acc"])) and agree >= int(0.85 * len(ids)) and vl["degenerate_count"] <= hf["degenerate_count"] + 1:
        print("PARITY HOLDS -> vLLM faithful & usable -> LOCK stack (vLLM "
              f"{vl.get('vllm','?')} + tf {vl['transformers']}).")
    elif hf["degenerate_count"] <= 2 and vl["degenerate_count"] >= int(0.5 * len(ids)):
        print("vLLM degenerates while HF clean on SAME pod/tf -> vLLM-SPECIFIC, not tf -> "
              "vLLM not usable for THIS checkpoint; revisit on v2.")
    elif hf["degenerate_count"] >= int(0.5 * len(ids)):
        print("BOTH degenerate -> fragile v1 checkpoint at this tf, not the backend -> "
              "re-test on well-trained v2.")
    else:
        print("Mixed/partial divergence -> inspect per-problem rows above before concluding.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["vllm", "hf"])
    ap.add_argument("--model", default="/workspace/model")
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    if args.compare:
        compare(); return
    probs = load_problems(args.n)
    blob = run_vllm(args.model, probs) if args.backend == "vllm" else run_hf(args.model, probs)
    blob = score(blob)
    out = OUT_DIR / f"parity_{args.backend}.json"
    json.dump(blob, open(out, "w"), indent=2)
    print(f"\n[{args.backend}] tf {blob['transformers']} "
          f"{'vllm '+blob.get('vllm','') if args.backend=='vllm' else ''} "
          f"acc {blob['acc']}/{blob['n']}  degenerate {blob['degenerate_count']}  -> {out}")


if __name__ == "__main__":
    main()
