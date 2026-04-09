#!/usr/bin/env python3
"""
Generate Qwen3-32B reasoning traces on MATH train set via OpenRouter.

Output: data/traces/qwen32b_math_traces.jsonl
Each line:
  {
    "id": int,
    "problem": str,
    "level": str,
    "subject": str,
    "expected": str,           # ground truth \boxed{...} answer
    "full_response": str,      # complete raw API response
    "thinking": str,           # content inside <think>...</think>
    "solution": str,           # content after </think>
    "predicted": str,          # extracted \boxed{} answer from solution
    "correct": bool            # predicted == expected (normalized)
  }

Summary: outputs/qwen32b_traces_summary.json
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
# Config
# ---------------------------------------------------------------------------

MODEL = "qwen/qwen3-32b"
BASE_URL = "https://openrouter.ai/api/v1"

# Qwen team recommended thinking-mode params (explicitly warns against temp=0)
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
MAX_TOKENS = 32768   # 32k covers 100% of observed trace lengths (p95 ~60k chars / 3.5 ≈ 17k tokens)
                     # 16384 caused 249/7490 truncations (thinking cut off, no \boxed{})

CONCURRENCY = 8      # parallel API requests
RETRY_LIMIT = 5
RETRY_BACKOFF_BASE = 2.0

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "traces"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TRACES_FILE = DATA_DIR / "qwen32b_math_traces.jsonl"
SUMMARY_FILE = OUTPUT_DIR / "qwen32b_traces_summary.json"
MATH_CACHE = PROJECT_ROOT / "data" / "math_train.jsonl"

# ---------------------------------------------------------------------------
# Answer extraction & normalization
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> str | None:
    """Extract content of last \\boxed{...} in text, handling nested braces."""
    pattern = r'\\boxed\{'
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None
    # Use the last match (most likely the final answer)
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


def normalize_answer(ans: str | None, strip_text_content: bool = True) -> str:
    """
    Normalize LaTeX math answers for equivalence comparison.

    Handles the main sources of false-negatives we observed in Qwen3-32B traces:
      - \\dfrac / \\tfrac → \\frac  (display/text fracs are identical)
      - \\frac14 / \\frac{1}4 → \\frac{1}{4}  (shorthand forms)
      - \\text{dogs} / \\mbox{sq ft} stripped  (units in word problems)
      - \\mathbf{N} / \\boldsymbol → N  (bold font, same value)
      - {,} thousands separator  (1{,}008 → 1008)
      - ^{\\circ} / \\circ stripped  (degree symbol)
      - \\%, $ stripped  (percent/currency)
      - LaTeX spacing macros stripped  (\\!, \\,, \\;, etc.)
      - \\phantom stripped  (alignment spacing)
      - \\left / \\right stripped  (bracket sizing)
      - All whitespace collapsed then removed
      - 2.50 → 2.5, 2.0 → 2  (trailing decimal zeros)
      - Trailing period stripped  (-\\frac{5}{3}. → -\\frac{5}{3})
      - Comma thousands separator  (1,234 → 1234, but NOT lists like -1,1)
      - Base notation suffix  (2677_9 → 2677)
      - Plain slash fractions  (-5/4 → \\frac{-5}{4})
      - Negative from numerator  (\\frac{-N}{D} → -\\frac{N}{D})
      - \\pi in numerator pulled out  (\\frac{9\\pi}{4} → \\frac{9}{4}\\pi)
      - Leading decimal  (.175 → 0.175)
      - \\sqrt shorthand  (\\sqrt2 → \\sqrt{2})

    NOT handled (document for human review):
      - Algebraic equivalence: 12(√3+1) vs 12+12√3, 1/√2 vs √2/2
      - Multi-choice letter answers (A/B/C/D/E) when expected is numeric
      - Partial-answer extraction (model gives only one of multiple required values)
      - Truncated responses (no \\boxed{} written due to token limit)
      - pmatrix \\\\ normalization edge cases
    """
    if ans is None:
        return ""
    s = ans.strip()

    # 1. Fraction variants → \frac
    s = s.replace(r'\dfrac', r'\frac').replace(r'\tfrac', r'\frac')

    # 2. Math font commands → content
    s = re.sub(r'\\math(?:bf|it|rm|bb|sf|tt|cal|scr)\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\\(?:boldsymbol|bm)\{([^}]*)\}', r'\1', s)

    # 3. \text / \mbox
    if strip_text_content:
        # Strip the wrapper AND content — units/words are noise for numeric problems
        s = re.sub(r'\\text\s*\{[^}]*\}', '', s)
        s = re.sub(r'\\mbox\s*\{[^}]*\}', '', s)
    else:
        # Preserve content for pure-text answers (day names, colors, etc.)
        s = re.sub(r'\\text\s*\{([^}]*)\}', lambda m: m.group(1), s)
        s = re.sub(r'\\mbox\s*\{([^}]*)\}', lambda m: m.group(1), s)

    # 4. {,} thousands separator
    s = re.sub(r'\{,\}', '', s)

    # 5. Degree symbol
    s = re.sub(r'\^\s*\{?\s*\\circ\s*\}?', '', s)
    s = s.replace(r'\circ', '')

    # 6. Percent / dollar
    s = s.replace(r'\%', '').replace('%', '')
    s = s.replace(r'\$', '').replace('$', '')

    # 7. LaTeX spacing macros
    for sp in [r'\!', r'\,', r'\;', r'\:', r'\quad', r'\qquad', r'\ ']:
        s = s.replace(sp, '')

    # 8. \phantom
    s = re.sub(r'\\phantom\{[^}]*\}', '', s)
    s = re.sub(r'\\phantom', '', s)

    # 9. \left / \right
    s = re.sub(r'\\(?:left|right)\s*', '', s)

    # 10. Collapse all whitespace (enables shorthand frac matching below)
    s = re.sub(r'\s+', '', s)

    # 11. Shorthand fractions (after whitespace removal for \frac 1 8 → \frac18)
    for _ in range(3):
        s = re.sub(r'\\frac(\d)(\d)', r'\\frac{\1}{\2}', s)
        s = re.sub(r'\\frac\{([^}]+)\}(\d)', r'\\frac{\1}{\2}', s)
        s = re.sub(r'\\frac(\d)\{([^}]+)\}', r'\\frac{\1}{\2}', s)
        s = re.sub(r'\\frac\{([^}]+)\}([a-zA-Z])', r'\\frac{\1}{\2}', s)
        s = re.sub(r'\\frac([a-zA-Z0-9])\{([^}]+)\}', r'\\frac{\1}{\2}', s)
        s = re.sub(r'\\frac([a-zA-Z0-9])([a-zA-Z0-9])', r'\\frac{\1}{\2}', s)

    # 12. \sqrt without braces: \sqrt2 → \sqrt{2}
    s = re.sub(r'\\sqrt([a-zA-Z0-9])', r'\\sqrt{\1}', s)

    # 13. Trailing decimal zeros: 2.50 → 2.5, 2.0 → 2
    s = re.sub(r'(\d+\.\d*[1-9])0+\b', r'\1', s)
    s = re.sub(r'(\d+)\.0+\b', r'\1', s)

    # 14. Trailing period (e.g. from base notation or sloppy formatting)
    s = re.sub(r'\.$', '', s)

    # 15. Comma thousands separator (only 3-digit groups to avoid eating comma-lists)
    s = re.sub(r'(\d),(\d{3})(?!\d)', lambda m: m.group(1) + m.group(2), s)

    # 16. Base notation suffix: 2677_9 → 2677
    s = re.sub(r'_(\d)$', '', s)

    # 17. Plain-slash numeric fractions → \frac
    s = re.sub(r'(-?\d+)/(-?\d+)', r'\\frac{\1}{\2}', s)

    # 18. Negative from \frac numerator: \frac{-N}{D} → -\frac{N}{D}
    s = re.sub(r'\\frac\{-([^}]+)\}', r'-\\frac{\1}', s)

    # 19. \pi in \frac numerator → pull outside: \frac{9\pi}{4} → \frac{9}{4}\pi
    def pull_pi_out(t: str) -> str:
        def repl(m: re.Match) -> str:
            num, den = m.group(1), m.group(2)
            if r'\pi' in num and r'\pi' not in den:
                num_clean = num.replace(r'\pi', '') or '1'
                return r'\frac{' + num_clean + '}{' + den + r'}\pi'
            return m.group(0)
        return re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', repl, t)
    s = pull_pi_out(s)

    # 20. Leading decimal: .175 → 0.175
    s = re.sub(r'^\.(\d)', r'0.\1', s)

    return s.strip()


# ---------------------------------------------------------------------------
# math-verify (primary evaluator)
# ---------------------------------------------------------------------------

try:
    from math_verify import parse as mv_parse, verify as mv_verify_fn
    _MATHVERIFY_AVAILABLE = True
except ImportError:
    _MATHVERIFY_AVAILABLE = False


def answers_match(predicted: str | None, expected: str | None) -> bool:
    """
    Check answer equivalence.

    Primary: math-verify (HuggingFace) — ANTLR4/SymPy symbolic comparison.
      Handles algebraic equivalence, sets/intervals, matrices, relations,
      units, unicode, and all LaTeX normalization edge cases.
      Install: pip install 'math-verify[antlr4_13_2]'

    Fallback: hand-rolled regex normalizer (if math-verify not installed).
      Two-pass: strip \\text{} content, then preserve it.
      Also strips variable assignments: "k = 6" → "6"
    """
    if not predicted or not expected:
        return False

    # Primary: math-verify
    # Wrap in $...$ so LatexExtractionConfig can parse bare LaTeX expressions
    # (e.g. \dfrac{3}{2}, t^7, \text{odd} all fail without a math environment wrapper)
    if _MATHVERIFY_AVAILABLE:
        try:
            gold = mv_parse(f"${expected}$")
            ans = mv_parse(f"${predicted}$")
            if gold and ans:
                return bool(mv_verify_fn(gold, ans))
        except Exception:
            pass  # fall through to regex normalizer on parse error

    # Fallback: regex normalizer
    def strip_var_assign(s: str) -> str:
        m = re.match(r'^[a-zA-Z]\s*=\s*(.+)$', s)
        return m.group(1) if m else s

    for strip_text in (True, False):
        ne = normalize_answer(expected, strip_text)
        np_ = normalize_answer(predicted, strip_text)
        if ne == np_ and ne != "":
            return True
        ne2 = strip_var_assign(ne)
        np2 = strip_var_assign(np_)
        if ne2 == np2 and ne2 != "":
            return True
        if ne2 == np_ and ne2 != "":
            return True
        if ne == np2 and ne != "":
            return True

    return False


# ---------------------------------------------------------------------------
# Trace parsing
# ---------------------------------------------------------------------------

def parse_response(full_response: str) -> tuple[str, str]:
    """Split full_response into (thinking, solution).
    thinking = content inside <think>...</think>
    solution = everything after </think>
    """
    think_match = re.search(r'<think>([\s\S]*?)</think>', full_response, re.IGNORECASE)
    if think_match:
        thinking = think_match.group(1).strip()
        solution = full_response[think_match.end():].strip()
    else:
        # No think tags — model responded directly (shouldn't happen with thinking on)
        thinking = ""
        solution = full_response.strip()
    return thinking, solution


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

async def call_api(
    client: httpx.AsyncClient,
    api_key: str,
    problem: str,
    semaphore: asyncio.Semaphore,
) -> str:
    """Call OpenRouter Qwen3-32B, return full response text."""
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
                                # /think soft-switch enables Qwen3 thinking mode via API
                                # (chat template enable_thinking=True equivalent)
                                "content": f"/think\n\nSolve the following math problem. Show your full reasoning, then give the final answer in \\boxed{{}}.\n\nProblem: {problem}",
                            }
                        ],
                        "temperature": TEMPERATURE,
                        "top_p": TOP_P,
                        "top_k": TOP_K,
                        "max_tokens": MAX_TOKENS,
                    },
                    timeout=120.0,
                )

                if resp.status_code == 429:
                    wait = RETRY_BACKOFF_BASE ** attempt
                    await asyncio.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()
                msg = data["choices"][0]["message"]
                content = msg.get("content", "") or ""
                # Some providers expose thinking in a separate field
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
# Dataset loading
# ---------------------------------------------------------------------------

def load_math_train() -> list[dict]:
    """Load MATH train set. Uses local cache if available."""
    if MATH_CACHE.exists():
        print(f"Loading from cache: {MATH_CACHE}")
        problems = []
        with open(MATH_CACHE) as f:
            for line in f:
                problems.append(json.loads(line))
        print(f"Loaded {len(problems)} problems from cache")
        return problems

    print("Downloading MATH train set from HuggingFace...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: datasets not installed. Run: pip install datasets")
        sys.exit(1)

    # EleutherAI/hendrycks_math is split by subject — load all 7 and combine
    SUBJECTS = [
        "algebra", "counting_and_probability", "geometry",
        "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
    ]
    all_items = []
    for subj in SUBJECTS:
        ds = load_dataset("EleutherAI/hendrycks_math", subj, split="train")
        all_items.extend(ds)
        print(f"  {subj}: {len(ds)} problems")
    print(f"Downloaded {len(all_items)} problems total")

    MATH_CACHE.parent.mkdir(parents=True, exist_ok=True)
    problems = []
    with open(MATH_CACHE, "w") as f:
        for i, item in enumerate(all_items):
            # Extract ground truth boxed answer from the solution
            expected = extract_boxed(item.get("solution", ""))
            record = {
                "id": i,
                "problem": item["problem"],
                "level": item.get("level", ""),
                "subject": item.get("type", ""),
                "expected": expected,
                "expected_solution": item.get("solution", ""),
            }
            problems.append(record)
            f.write(json.dumps(record) + "\n")

    print(f"Cached to {MATH_CACHE}")
    return problems


# ---------------------------------------------------------------------------
# Resume logic
# ---------------------------------------------------------------------------

def load_completed_ids() -> set[int]:
    """Return set of problem IDs already in the traces file."""
    if not TRACES_FILE.exists():
        return set()
    completed = set()
    with open(TRACES_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    completed.add(record["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return completed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def process_problem(
    client: httpx.AsyncClient,
    api_key: str,
    problem: dict,
    semaphore: asyncio.Semaphore,
    out_file,
    lock: asyncio.Lock,
    stats: dict,
) -> None:
    try:
        full_response = await call_api(client, api_key, problem["problem"], semaphore)
        thinking, solution = parse_response(full_response)
        predicted = extract_boxed(solution) or extract_boxed(full_response)
        correct = answers_match(predicted, problem["expected"])

        record = {
            "id": problem["id"],
            "problem": problem["problem"],
            "level": problem["level"],
            "subject": problem["subject"],
            "expected": problem["expected"],
            "full_response": full_response,
            "thinking": thinking,
            "solution": solution,
            "predicted": predicted,
            "correct": correct,
        }

        async with lock:
            out_file.write(json.dumps(record) + "\n")
            out_file.flush()
            stats["done"] += 1
            if correct:
                stats["correct"] += 1

    except Exception as e:
        async with lock:
            stats["errors"] += 1
        print(f"\nERROR problem {problem['id']}: {e}", file=sys.stderr)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=None, help="OpenRouter API key (default: read from openclaw.json)")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--limit", type=int, default=None, help="Process only first N problems (for testing)")
    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        # Try reading from openclaw config
        config_path = Path.home() / ".openclaw" / "openclaw.json"
        if config_path.exists():
            import re as _re
            raw = config_path.read_text()
            match = _re.search(r'"openrouter"[\s\S]{0,300}?"apiKey":\s*"(sk-or-[^"]+)"', raw)
            if match:
                api_key = match.group(1)
        if not api_key:
            print("ERROR: No OpenRouter API key found. Pass --api-key or set OPENROUTER_API_KEY.", file=sys.stderr)
            sys.exit(1)

    print(f"API key: {api_key[:20]}...")

    # Load dataset
    problems = load_math_train()
    if args.limit:
        problems = problems[:args.limit]
    print(f"Total problems: {len(problems)}")

    # Resume
    completed = load_completed_ids()
    todo = [p for p in problems if p["id"] not in completed]
    print(f"Already done: {len(completed)} | Remaining: {len(todo)}")

    if not todo:
        print("All done!")
        return

    # Setup output
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stats = {"done": len(completed), "correct": 0, "errors": 0}
    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(args.concurrency)

    # Count existing correct from completed (for accurate accuracy tracking)
    if TRACES_FILE.exists():
        with open(TRACES_FILE) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("correct"):
                        stats["correct"] += 1
                except:
                    pass

    start_time = time.time()

    async with httpx.AsyncClient() as client:
        with open(TRACES_FILE, "a") as out_file:
            tasks = [
                process_problem(client, api_key, p, semaphore, out_file, lock, stats)
                for p in todo
            ]
            # Process with progress bar
            for coro in atqdm.as_completed(tasks, total=len(tasks), desc="Traces"):
                await coro

                # Print running accuracy every 100
                if stats["done"] % 100 == 0 and stats["done"] > 0:
                    acc = stats["correct"] / stats["done"] * 100
                    elapsed = time.time() - start_time
                    rate = (stats["done"] - len(completed)) / elapsed * 60
                    print(f"\n[{stats['done']}/{len(problems)}] acc={acc:.1f}% | {rate:.0f} problems/min | errors={stats['errors']}")

    # Final summary
    total_done = stats["done"]
    accuracy = stats["correct"] / total_done if total_done > 0 else 0

    # Per-level and per-subject breakdown
    level_stats: dict[str, dict] = {}
    subject_stats: dict[str, dict] = {}

    with open(TRACES_FILE) as f:
        for line in f:
            try:
                r = json.loads(line)
                lvl = r.get("level", "unknown")
                subj = r.get("subject", "unknown")
                for key, d in [(lvl, level_stats), (subj, subject_stats)]:
                    if key not in d:
                        d[key] = {"correct": 0, "total": 0}
                    d[key]["total"] += 1
                    if r.get("correct"):
                        d[key]["correct"] += 1
            except:
                pass

    summary = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "total": total_done,
        "correct": stats["correct"],
        "accuracy": round(accuracy, 4),
        "errors": stats["errors"],
        "per_level": {
            k: {"correct": v["correct"], "total": v["total"], "accuracy": round(v["correct"]/v["total"], 4)}
            for k, v in sorted(level_stats.items())
        },
        "per_subject": {
            k: {"correct": v["correct"], "total": v["total"], "accuracy": round(v["correct"]/v["total"], 4)}
            for k, v in sorted(subject_stats.items())
        },
    }

    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE: {total_done} problems | Accuracy: {accuracy:.1%} | Errors: {stats['errors']}")
    print(f"Traces: {TRACES_FILE}")
    print(f"Summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
