#!/usr/bin/env python3
"""
10_rerun_truncated.py — Re-run truncated traces at MAX_TOKENS=32768.

Identifies traces where thinking was cut off (no \\boxed{} in solution),
re-generates them via OpenRouter at 32k token limit, and patches the
rescored JSONL in place.

Background: 249/7490 traces in the original run were truncated because
MAX_TOKENS=16384 was too low. Median truncated thinking: 44k chars (p95: 60k).
At 32k tokens x 3.5 chars/token ≈ 112k chars — covers all observed lengths.

Usage:
    python scripts/10_rerun_truncated.py
    python scripts/10_rerun_truncated.py --dry-run  # just count truncated, don't rerun
    python scripts/10_rerun_truncated.py --limit 10  # test with 10 problems
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx
from tqdm.asyncio import tqdm as atqdm

# ---------------------------------------------------------------------------
# Config — same as Script 07 but 32k tokens
# ---------------------------------------------------------------------------

MODEL = "qwen/qwen3-32b"
BASE_URL = "https://openrouter.ai/api/v1"
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
MAX_TOKENS = 32768   # 2x original; covers p95 of observed truncated traces

CONCURRENCY = 8
RETRY_LIMIT = 5
RETRY_BACKOFF_BASE = 2.0

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "traces"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Input: use the rescored file (most up-to-date correct flags)
INPUT_FILE = DATA_DIR / "qwen32b_math_traces_rescored.jsonl"
# Output: patched version with rerun traces merged in
OUTPUT_FILE = DATA_DIR / "qwen32b_math_traces_rerun.jsonl"

# ---------------------------------------------------------------------------
# Answer extraction (same as Script 07)
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> str | None:
    pattern = r'\\boxed\{'
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    if depth == 0:
        return text[start:i-1].strip()
    return None


def parse_response(full_response: str) -> tuple[str, str]:
    think_match = re.search(r'<think>([\s\S]*?)</think>', full_response, re.IGNORECASE)
    if think_match:
        thinking = think_match.group(1).strip()
        solution = full_response[think_match.end():].strip()
    else:
        thinking = ""
        solution = full_response.strip()
    return thinking, solution


# ---------------------------------------------------------------------------
# math-verify (primary evaluator, matches Script 07 / Script 09)
# ---------------------------------------------------------------------------

try:
    from math_verify import parse as mv_parse, verify as mv_verify_fn
    _MATHVERIFY_AVAILABLE = True
except ImportError:
    _MATHVERIFY_AVAILABLE = False


def answers_match_mv(predicted: str | None, expected: str | None) -> bool:
    if not predicted or not expected:
        return False
    if _MATHVERIFY_AVAILABLE:
        try:
            gold = mv_parse(f"${expected}$")
            ans = mv_parse(f"${predicted}$")
            if gold and ans:
                return bool(mv_verify_fn(gold, ans))
        except Exception:
            pass
    return False


# ---------------------------------------------------------------------------
# Identify truncated traces
# ---------------------------------------------------------------------------

def find_truncated(traces: list[dict]) -> list[dict]:
    """
    A trace is truncated if:
      - solution has no \\boxed{} (thinking was cut off before answer)
      - thinking is very long AND solution is short/empty
    """
    truncated = []
    for t in traces:
        solution = t.get("solution", "")
        thinking = t.get("thinking", "")
        has_boxed = bool(extract_boxed(solution)) or bool(extract_boxed(t.get("full_response", "")))
        if not has_boxed:
            truncated.append(t)
    return truncated


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

async def call_api(
    client: httpx.AsyncClient,
    api_key: str,
    problem: str,
    semaphore: asyncio.Semaphore,
) -> str:
    async with semaphore:
        for attempt in range(RETRY_LIMIT):
            try:
                resp = await client.post(
                    f"{BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/clawd/qwen3-gsm8k-demo",
                    },
                    json={
                        "model": MODEL,
                        "messages": [
                            {
                                "role": "user",
                                "content": f"/think\n\nSolve the following math problem. Show your full reasoning, then give the final answer in \\boxed{{}}.\n\nProblem: {problem}",
                            }
                        ],
                        "temperature": TEMPERATURE,
                        "top_p": TOP_P,
                        "top_k": TOP_K,
                        "max_tokens": MAX_TOKENS,
                    },
                    timeout=180.0,  # longer timeout for 32k generation
                )

                if resp.status_code == 429:
                    wait = RETRY_BACKOFF_BASE ** attempt
                    await asyncio.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()
                msg = data["choices"][0]["message"]
                content = msg.get("content", "") or ""
                reasoning = msg.get("reasoning", "") or msg.get("reasoning_content", "") or ""
                if reasoning:
                    return f"<think>{reasoning}</think>\n{content}"
                return content

            except (httpx.TimeoutException, httpx.RemoteProtocolError) as e:
                if attempt < RETRY_LIMIT - 1:
                    wait = RETRY_BACKOFF_BASE ** attempt
                    await asyncio.sleep(wait)
                else:
                    raise RuntimeError(f"API call failed after {RETRY_LIMIT} attempts: {e}")

        raise RuntimeError(f"API call failed after {RETRY_LIMIT} attempts (rate limit)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def rerun_one(
    client: httpx.AsyncClient,
    api_key: str,
    record: dict,
    semaphore: asyncio.Semaphore,
    stats: dict,
    lock: asyncio.Lock,
) -> dict:
    try:
        full_response = await call_api(client, api_key, record["problem"], semaphore)
        thinking, solution = parse_response(full_response)
        predicted = extract_boxed(solution) or extract_boxed(full_response)
        correct = answers_match_mv(predicted, record["expected"])

        updated = dict(record)
        updated.update({
            "full_response": full_response,
            "thinking": thinking,
            "solution": solution,
            "predicted": predicted,
            "correct": correct,
            "correct_mathverify": correct,
            "rerun": True,
            "max_tokens_used": MAX_TOKENS,
        })

        async with lock:
            stats["done"] += 1
            if correct:
                stats["correct"] += 1
        return updated

    except Exception as e:
        async with lock:
            stats["errors"] += 1
        print(f"\nERROR problem {record['id']}: {e}", file=sys.stderr)
        return record  # keep original on error


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--limit", type=int, default=None, help="Rerun only first N truncated")
    parser.add_argument("--dry-run", action="store_true", help="Just count truncated, don't call API")
    parser.add_argument("--input", default=str(INPUT_FILE))
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    # Load all traces
    traces = []
    with open(input_path) as f:
        for line in f:
            traces.append(json.loads(line))
    print(f"Loaded {len(traces)} traces from {input_path.name}")

    # Find truncated
    truncated = find_truncated(traces)
    print(f"Truncated (no \\boxed{{}}): {len(truncated)}")

    if args.dry_run:
        for t in truncated[:20]:
            print(f"  id={t['id']} level={t['level']} subject={t['subject']}")
        if len(truncated) > 20:
            print(f"  ... and {len(truncated)-20} more")
        return

    if not truncated:
        print("No truncated traces found. All good!")
        return

    if args.limit:
        truncated = truncated[:args.limit]
        print(f"Limited to {len(truncated)} reruns")

    # Get API key
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        config_path = Path.home() / ".openclaw" / "openclaw.json"
        if config_path.exists():
            raw = config_path.read_text()
            match = re.search(r'"openrouter"[\s\S]{0,300}?"apiKey":\s*"(sk-or-[^"]+)"', raw)
            if match:
                api_key = match.group(1)
    if not api_key:
        print("ERROR: No OpenRouter API key found.", file=sys.stderr)
        sys.exit(1)

    print(f"\nRe-running {len(truncated)} traces at MAX_TOKENS={MAX_TOKENS}...")

    # Build id→record map for patching
    id_to_record = {t["id"]: t for t in traces}
    truncated_ids = {t["id"] for t in truncated}

    stats = {"done": 0, "correct": 0, "errors": 0}
    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(args.concurrency)
    updated_records: dict[int, dict] = {}

    start = time.time()
    async with httpx.AsyncClient() as client:
        tasks = [
            rerun_one(client, api_key, t, semaphore, stats, lock)
            for t in truncated
        ]
        for coro in atqdm.as_completed(tasks, total=len(tasks), desc="Rerun"):
            result = await coro
            updated_records[result["id"]] = result

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.0f}s | {stats['done']} rerun | {stats['correct']} correct | {stats['errors']} errors")

    # Write merged output: original traces with truncated ones patched in
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    total_correct = 0
    with open(OUTPUT_FILE, "w") as f:
        for t in traces:
            tid = t["id"]
            if tid in truncated_ids and tid in updated_records:
                record = updated_records[tid]
            else:
                record = t
            if record.get("correct"):
                total_correct += 1
            f.write(json.dumps(record) + "\n")

    print(f"\nMerged output → {OUTPUT_FILE}")
    print(f"Total correct: {total_correct}/{len(traces)} ({total_correct/len(traces):.2%})")
    print(f"\nNext step: run Script 09 on this file:")
    print(f"  python scripts/09_rescore_mathverify.py --input {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
