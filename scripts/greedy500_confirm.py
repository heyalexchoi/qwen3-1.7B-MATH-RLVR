import sys, os, json, time, subprocess, hashlib, datetime, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/workspace/qwen3-math-rlvr/scripts")
sys.path.insert(0, "/workspace/qwen3-math-rlvr")
import torch
from transformers import AutoModelForCausalLM
import math500_eval as M
from prompts import create_prompt
from datasets import load_dataset
from math_verify import parse, verify

MODEL="heyalexchoi/qwen3-1.7b-math-grpo"; REV="63870ec239b2"
tok = M.load_tokenizer_safe(MODEL, revision=REV)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()
ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
rows = list(ds)
print(f"loaded {len(rows)} problems; commit check via M.STOP_STRINGS={M.STOP_STRINGS} max_new={M.MAX_NEW_TOKENS}", flush=True)

B=25; results=[]; t0=time.time()
for s in range(0, len(rows), B):
    chunk = rows[s:s+B]
    prompts = [create_prompt(r["problem"]) for r in chunk]
    outs = M.generate_batch(model, tok, prompts, n_per_prompt=1, temperature=0.0, max_new_tokens=M.MAX_NEW_TOKENS)
    for r, o in zip(chunk, outs):
        results.append({"problem":r["problem"],"expected":r["answer"],"level":r["level"],"subject":r["subject"],"response":o[0]})
    print(f"  {s+len(chunk)}/{len(rows)}  ({time.time()-t0:.0f}s)", flush=True)

c=0; by={}
for r in results:
    ok=False
    try: ok=bool(verify(parse(f"${r['expected']}$"), parse(r["response"])))
    except: pass
    r["correct"]=ok; c+=ok
    L=str(r["level"]); by.setdefault(L,[0,0]); by[L][0]+=ok; by[L][1]+=1
acc=c/len(results)
def git(*a):
    try: return subprocess.check_output(["git","-C","/workspace/qwen3-math-rlvr",*a],stderr=subprocess.DEVNULL).decode().strip()
    except: return None
out={"model":MODEL,"revision":REV,"backend":"HF-generate","mode":"greedy-only-full500",
     "greedy_pass1":acc,"correct":c,"total":len(results),
     "by_level":{k:{"correct":v[0],"total":v[1],"acc":round(v[0]/v[1],4)} for k,v in sorted(by.items())},
     "provenance":{"git_commit":git("rev-parse","HEAD"),"git_describe":git("describe","--tags","--always","--dirty"),
                   "run_utc":datetime.datetime.utcnow().isoformat()+"Z","torch":torch.__version__},
     "results":results}
json.dump(out, open("/workspace/qwen3-math-rlvr/outputs/grpo3000_greedy500_confirm.json","w"), indent=1)
print(f"\nGREEDY pass@1 (full 500, HF, rev {REV}): {c}/{len(results)} = {acc:.4f}")
print("by level:", {k:f"{v[0]}/{v[1]}={v[0]/v[1]:.2f}" for k,v in sorted(by.items())})
