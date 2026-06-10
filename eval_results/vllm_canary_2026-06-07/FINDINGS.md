# vLLM canary findings — 2026-06-07 (pod 6t59eo45pi99ij, A40, driver CUDA 12.8)

*Written by Claude Opus 4.8 — 2026-06-07.*

**Question:** can a MODERN vLLM run the SFT checkpoint cleanly? (The old "vLLM downgrades
transformers and breaks it" was suspected to be a version artifact.)

## 1. Version confound — DEAD
Modern vLLM **0.22.1 AND 0.11.0 both pull transformers 5.10.2**, NOT the broken 4.51.3.
The "vLLM downgrades tf and breaks the model" story was purely an artifact of pinning
**ancient vLLM 0.8.5** (Apr 2025), whose ceiling is tf 4.51.3.

## 2. Tokenizer config is MALFORMED in heyalexchoi/qwen3-1.7b-math-sft
- `extra_special_tokens` = **13-item LIST** (modern tf expects a dict → `.keys()` crash on
  tf 4.56; tf 5.x removed `all_special_tokens_extended` → a different crash)
- `added_tokens_decoder` = **EMPTY (0)** vs stock Qwen3-1.7B-Base = **26**

→ a fragile config some tf versions tolerate (5.5.3) and others choke on, in different ways.
**ACTION for SFT v2: save the tokenizer canonically** (copy stock tokenizer files).

## 3. Numerics (the real test)
With the tokenizer **patched** (`extra_special_tokens`→`{}`, `chat_template` restored) +
vLLM 0.11.0 + tf 4.56.2 + torch 2.8/cu128:

| run | result |
|---|---|
| greedy 572 | **0/1** — 8170 tok, `finish=length`, no `\boxed` |
| sample8 572 | **0/8** — all 8170 tok, `finish=length` |

vs blessed **HF + tf5.5.3**: greedy clean `\boxed{9}` @1787 tok, 6/8 samples clean.

**Conclusion:** the checkpoint collapses under modern vLLM **even with the tokenizer fixed**.
- Ruled OUT as the cause: tokenizer config; ancient-vLLM-version.
- Still entangled: vLLM-backend vs tf-version (clean isolator = HF @ tf4.51/4.56, not run).
- **Practical:** vLLM is NOT a usable pass@8 shortcut for THIS checkpoint. HF+tf5.5.3 only.

**Constraint:** this pod's CUDA-12.8 driver caps vLLM at ~0.11 (latest 0.22 needs CUDA 13 /
cu130 torch). "Make latest vLLM work" = a newer-driver pod, not more pinning.

## 4. TERMINATION DISCRIMINATOR — DEFINITIVE (2026-06-08, A40 pod um1p0k29i2f5cx)

*Written by Claude Opus 4.8 — 2026-06-08.* Ran the master discriminator the earlier session
never did: vLLM 0.11.0 + tf **4.56.2** (version-aligned, no load crash) + the SFT checkpoint
with a **fully repaired tokenizer** (`fix_sft_tokenizer.py`: added_tokens_decoder rebuilt from
tokenizer.json's 26 tokens, malformed extra_special_tokens dropped, chat_template embedded),
dumping output **token_ids** (`vllm_canary_tokendump.py`).

| run | result |
|---|---|
| greedy 572 | ntok=8192 finish=length, boxed=False, **emits eos id 151645? NO, emits </think> id 151668? NO** |
| sample0–7 | **all** ntok=8192 finish=length, never emit eos/</think> |

Greedy text tail = `616161616161…` (degenerate repetition — the non-termination disease).

Greedy result: never emits a stop token, degenerates to `616161…`. BUT see the correction below —
this run had a **sampling-param bug**, so it does NOT establish a numerics/backend verdict.

**CORRECTION (same day): the canary used the WRONG sampling params.** The blessed HF greedy run
(40.2%, clean `\boxed{9}` @1787 tok) uses the chat `FORMAT_DEFAULTS`: **`repetition_penalty=1.05`**.
This token-dump canary passed `SamplingParams(temperature=0.0, max_tokens=8192)` — **no repetition
penalty (1.0)**. A fragile undertrained model decoded greedy with *no* penalty looping on `616161…`
is the textbook outcome and is **plausibly unrelated to vLLM**. So:

## 5. ISOLATED ROOT CAUSE — it's the TRANSFORMERS VERSION, not vLLM (2026-06-08, pod kbwj8w8cxretlv)

Ran the two corrected experiments the earlier verdict was missing. **rep_penalty refuted, then a clean
backend-vs-tf isolation:**

| stack (572 greedy, rep_penalty 1.05) | result |
|---|---|
| HF + **tf 5.5.3** (blessed eval) | clean `\boxed{9}` @1787 tok ✓ |
| HF + **tf 4.56.2** (isolator, this pod) | degenerate `#\n#\n#\n…` loop, 8192 tok ✗ |
| vLLM 0.11.0 + tf 4.56.2 | degenerate `616161…` loop, 8192 tok ✗ |

**Conclusion:** the checkpoint generates correctly **only under transformers 5.5.3** (the version it
was saved with — `generation_config.json: transformers_version 5.5.3`). transformers **4.56.2 breaks
it regardless of backend** (HF degenerates too, just with a different junk token). So the vLLM collapse
was never about vLLM's kernels, the tokenizer, or repetition penalty — it's a **transformers-version
sensitivity**, and vLLM 0.11.0 is *pinned* to tf ≤ 4.56 (tf 5.x removed `all_special_tokens_extended`,
which vLLM 0.11 calls), so it is forced onto the version that breaks this model.

**What this corrects:** every prior "vLLM numerics / vLLM unusable" claim — over-conclusions. The real
blocker is the tf pin. **The open path to a USABLE vLLM:** a vLLM new enough to run tf 5.x (e.g. 0.22
pulls tf 5.10) — needs a CUDA-13/cu130 driver pod (driver ≥580; this A40 had 570). NOT YET TESTED.

**Practical today:** HF + tf 5.5.3 is the safe v2 eval path. **For v2:** the fragility may be specific
to this undertrained checkpoint and/or its saved config — re-test a well-trained v2 across tf versions
before assuming. Artifacts: `vllm_canary_tokendump.json` (rep 1.0), `vllm_canary_tokendump_rep105.json`
(rep 1.05). Both pods torn down.

## 6. RESOLVED — WORKING vLLM STACK FOUND & PINNED (2026-06-10, pod maenpd1vh3c3gp)

*Written by Claude Opus 4.8 — 2026-06-10.* The mission was: find a vLLM/tf combo that actually runs
the SFT checkpoint cleanly, then pin it. **Done — current vLLM works.**

**The pinned, verified-working stack** (A40, driver 570.211 / CUDA 12.8):

| package | version |
|---|---|
| vllm | **0.22.1 (cu129 wheel)** |
| transformers | **5.10.2** |
| torch | **2.11.0+cu129** |
| tokenizers | 0.22.2 |
| accelerate | 1.13.0 |
| numpy | 2.1.2 |

**Result (25 MATH-500 problems, greedy, rep_penalty 1.05):** vLLM **acc 17/25**, terminates cleanly
on 19/25 with correct well-formed `\boxed{}` answers at normal token counts (1–4k). The 6 failures
are `ntok=8192` pegs — **this v1 checkpoint's own known termination disease** (~53% peg rate measured
earlier), NOT a vLLM artifact. This is categorically different from the broken-stack collapse (§3–5),
where ALL generations pegged at 8192 emitting `616161…` garbage and scored 0.

**Why the earlier sessions failed to find this — the two real gotchas (both now navigated):**
1. **tf5-capable vLLM (≥0.20) pins torch 2.11, which is CUDA-13 era.** The DEFAULT `pip install vllm`
   pulls the **cu130** wheel (torch 2.11.0+cu130) → dies on a 12.8 driver (`driver too old, found
   12080`). The fix is the **cu129 wheel** — `torch 2.11.0+cu129` runs fine on the 570/CUDA-12.8 driver
   via CUDA minor-version forward-compat. So modern vLLM runs on the ordinary A40 pods we already use;
   you just must install the cu129 build, not the default. (No cu128 wheel exists for 0.20–0.22.)
2. Two install-time snags on the base `runpod/pytorch:2.8.0-cuda12.8.1` image: the stock **torchvision
   is built for torch 2.8** and crashes transformers' import (`operator torchvision::nms does not
   exist`) → `pip uninstall torchvision` (text models don't need it). And HF `device_map` needs
   **accelerate** → `pip install accelerate`.

**Install recipe** (see `docs/vllm-stack-pin.md`):
```
pip install "https://github.com/vllm-project/vllm/releases/download/v0.22.1/vllm-0.22.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
    --extra-index-url https://download.pytorch.org/whl/cu129
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu129   # force cu129, not cu130
pip uninstall -y torchvision        # mismatched with torch 2.11
pip install accelerate
```

**This corrects every prior "vLLM unusable / numerics" verdict in §3–5.** The blocker was never vLLM's
kernels — it was (a) ancient vLLM 0.11 force-downgrading tf to the version that breaks this checkpoint,
and (b) the default cu130 wheel being driver-incompatible. The checkpoint loads AND generates cleanly
under tf 5.10.2 (no tokenizer/`all_special_tokens_extended` crash).

**Cross-check vs blessed HF greedy** (same 25 problems, vs `sft_local_chat_max8192_math500_mv_rescored.json`
`pass1_greedy`): HF **15/25**, vLLM **17/25**, per-problem agreement **19/25**. Aggregate accuracy is
comparable (vLLM +2). Of the 6 disagreements: vLLM lost 2 (both its own 8192 pegs where HF terminated),
won 4 (HF greedy missed them). NOTE this cross-check varies TWO things at once (backend HF→vLLM AND tf
5.5.3→5.10.2) and the undertrained checkpoint sits at the cap, so near-boundary numeric diffs flip
individual problems — not a clean backend-only parity, but accuracy is on par and vLLM's losses are the
checkpoint's pegging, not mis-decoding.

**Caveat (honest):** per the user's call we did NOT run a same-pod, same-tf HF control this round — so this
is "vLLM produces correct, terminating output with HF-comparable aggregate accuracy," not a proven
bit-for-bit HF parity. The 6 pegs are consistent with the checkpoint's measured peg disease, not new vLLM
degeneration. **v2 plan:** train,
save, eval, and GRPO-rollout all under this one stack (tf 5.10.2 + vLLM 0.22.1+cu129) so the v2
checkpoint is born vLLM-compatible. Artifact: `eval_results/vllm_parity_2026-06-10/parity_vllm.json`.
Pod torn down; balance $32.72, no pods remain.
