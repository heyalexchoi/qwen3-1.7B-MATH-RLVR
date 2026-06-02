#!/usr/bin/env python3
"""No-GPU seam test for the unified scripts/math500_eval.py.

Covers everything EXCEPT model generation (which needs a GPU + the canary on the
next live run): helpers, both prompt formats, stop-id resolution, the per-sample
JSONL writer, and the pass1/pass8 schema contract with rescore_math500.py.

Run:  python3 tests/test_eval_seam.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import math500_eval as M  # noqa: E402

SFT_CKPT = ROOT / "outputs" / "sft_checkpoint"
PASS = []
FAIL = []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))


def test_extract_boxed():
    check("extract_boxed simple", M.extract_boxed(r"answer is \boxed{42}.") == "42")
    check("extract_boxed nested", M.extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}")
    check("extract_boxed last wins", M.extract_boxed(r"\boxed{1} then \boxed{2}") == "2")
    check("extract_boxed none", M.extract_boxed("no box here") == "")


def test_format_defaults():
    c, ch = M.FORMAT_DEFAULTS["completion"], M.FORMAT_DEFAULTS["chat"]
    check("completion default 2048", c["max_new_tokens"] == 2048)
    check("chat default 8192", ch["max_new_tokens"] == 8192)
    check("chat default temp 0.6", ch["temperature"] == 0.6)
    check("chat default rep_penalty 1.05", ch["repetition_penalty"] == 1.05)


def test_prompts_and_stop_ids():
    if not SFT_CKPT.exists():
        print(f"  [SKIP] tokenizer tests — {SFT_CKPT} not present")
        return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(SFT_CKPT))

    ds = [{"problem": "What is 2+2?", "answer": "4", "level": 1, "subject": "Algebra"}]
    comp = M.build_prompts(ds, "completion", None)
    chat = M.build_prompts(ds, "chat", tok)
    check("completion prompt is few-shot (has 'Problem:')", "Problem:" in comp[0])
    check("chat prompt uses chat template (has <|im_start|>)", "<|im_start|>" in chat[0])
    check("formats differ", comp[0] != chat[0])

    stop = M.get_stop_ids(tok)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    check("get_stop_ids includes <|im_end|> (151645)", im_end in stop, f"stop={stop}")
    check("get_stop_ids all ints >=0", all(isinstance(i, int) and i >= 0 for i in stop), f"stop={stop}")


def test_sample_writer():
    ex = {"unique_id": "test/algebra/1", "problem": "p?", "answer": "4",
          "level": 1, "subject": "Algebra"}
    with tempfile.TemporaryDirectory() as d:
        jp = Path(d) / "samples.jsonl"
        M._write_sample_rows(str(jp), ex, 0, "greedy",
                             [(r"the answer is \boxed{4}", 7)], 8192, "chat", "test-model")
        rows = [json.loads(line) for line in jp.read_text().splitlines()]
    required = {"unique_id", "pass_type", "sample_idx", "problem", "expected", "level",
                "subject", "response", "predicted", "n_tokens", "max_new_tokens", "format", "model"}
    check("jsonl row has all required keys", required <= set(rows[0]), f"missing={required - set(rows[0])}")
    check("jsonl predicted extracted", rows[0]["predicted"] == "4")
    check("jsonl n_tokens preserved", rows[0]["n_tokens"] == 7)
    if M.MATH_VERIFY_AVAILABLE:
        check("jsonl correct scored true for 4==4", rows[0].get("correct") is True)


def test_rescore_schema_contract():
    """A synthetic combined json in the pass1/pass8 schema must rescore cleanly."""
    data = {
        "model": "test", "format": "chat", "max_new_tokens": 8192,
        "dataset_stats": {"total": 2},
        "results": [
            {"problem": "p1", "expected": "4", "level": 1, "subject": "Algebra",
             "pass1": [{"response": r"\boxed{4}", "predicted": "4", "n_tokens": 3}],
             "pass8": [{"response": r"\boxed{4}", "predicted": "4", "n_tokens": 3},
                       {"response": r"\boxed{5}", "predicted": "5", "n_tokens": 3}]},
            {"problem": "p2", "expected": "10", "level": 5, "subject": "Geometry",
             "pass1": [{"response": r"\boxed{9}", "predicted": "9", "n_tokens": 3}],
             "pass8": [{"response": r"\boxed{10}", "predicted": "10", "n_tokens": 3}]},
        ],
    }
    with tempfile.TemporaryDirectory() as d:
        inp = Path(d) / "x_results.json"
        outp = Path(d) / "x_mv_rescored.json"
        inp.write_text(json.dumps(data))
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "rescore_math500.py"),
             "--input", str(inp), "--output", str(outp)],
            capture_output=True, text=True,
        )
        ok = r.returncode == 0 and outp.exists()
        check("rescore_math500.py consumes unified schema", ok, r.stderr[-300:])
        if ok:
            scored = json.loads(outp.read_text())
            s = scored["summary"]
            # p1: greedy 4==4 correct; p2: greedy 9!=10 wrong -> greedy 1/2
            check("rescore greedy pass@1 = 1/2", s["pass1_greedy"] == 0.5, f"got {s['pass1_greedy']}")
            # p1 pass8 has a 4 -> correct; p2 pass8 has 10 -> correct -> 2/2
            check("rescore pass@8 = 2/2", s["pass8"] == 1.0, f"got {s['pass8']}")


def main():
    print("== seam test: scripts/math500_eval.py (no GPU) ==")
    test_extract_boxed()
    test_format_defaults()
    test_prompts_and_stop_ids()
    test_sample_writer()
    test_rescore_schema_contract()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", ", ".join(FAIL))
        sys.exit(1)
    print("ALL PASS — generation paths still require the GPU canary (572≈8/8) on the live run.")


if __name__ == "__main__":
    main()
