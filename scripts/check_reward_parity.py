#!/usr/bin/env python3
"""
Pre-GRPO verification — Check 1: Reward function parity with baseline eval.

Confirms the GRPO reward function agrees with the baseline eval scorer on what
counts as correct. Catches silent disagreement between eval-time scoring and
training-time reward, which is a top failure mode for zero-RL runs.

Uses the EXACT reward function code path from grpo_train.py (same imports,
same extract_boxed, same math-verify calls, same regex fallback).

Inputs:
  outputs/baseline_math500_mv_rescored.json — 500 problems × 8 pass@8 rollouts

Procedure:
  1. Load all 4000 pass@8 rollouts.
  2. Determine ground-truth correct/incorrect by running math-verify directly
     on the already-extracted `predicted` vs `expected` from the baseline file.
     (This is the reference scorer: what the baseline eval determined was correct.)
  3. For each rollout, call the GRPO reward function with (response_text, expected)
     and record the returned reward.
  4. Compare reward function output to ground-truth labels.

Pass criteria:
  - All ground-truth-correct rollouts return reward=1.0 (0 misses allowed).
  - All ground-truth-incorrect rollouts return reward=0.0 (0 false positives allowed).

On failure, prints the first 5 mismatches: response text, expected, extracted
predicted, what math-verify said, what the reward function returned.

Usage (on pod):
  cd /workspace/qwen3-math-rlvr
  python scripts/check_reward_parity.py
  python scripts/check_reward_parity.py --data outputs/baseline_math500_mv_rescored.json
  python scripts/check_reward_parity.py --n_incorrect 200  # larger incorrect sample
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Import reward function directly from grpo_train — same code path as training
# ---------------------------------------------------------------------------

_scripts_dir = Path(__file__).parent
sys.path.insert(0, str(_scripts_dir))

try:
    from grpo_train import (
        extract_boxed,
        _mv_correct,
        correctness_reward,
        MATH_VERIFY_AVAILABLE,
    )
except ImportError as e:
    print(f"ERROR: Could not import from grpo_train.py: {e}")
    print("Make sure you are running from the repo root and dependencies are installed.")
    sys.exit(1)

# Reference scorer — math-verify directly on pre-extracted predicted vs expected.
# This mirrors what the baseline eval used (minus the fallback path) but is
# independent of extract_boxed so we can isolate any extraction differences.
try:
    from math_verify import parse as _ref_parse, verify as _ref_verify

    def reference_correct(predicted: str, expected: str) -> bool:
        """Math-verify on pre-extracted predicted (no extract_boxed call)."""
        if not predicted or not expected:
            return False
        try:
            gold = _ref_parse(f"${expected}$")
            ans = _ref_parse(f"${predicted}$")
            if gold and ans:
                return bool(_ref_verify(gold, ans))
        except Exception:
            pass
        return False

    REF_AVAILABLE = True

except ImportError:
    REF_AVAILABLE = False

    def reference_correct(predicted: str, expected: str) -> bool:  # type: ignore[misc]
        # Regex fallback — less accurate but still useful for catching gross mismatches
        from grpo_train import _regex_match
        return _regex_match(predicted, expected)


def load_rollouts(data_path: str) -> list[dict]:
    """Load all pass@8 rollouts from the baseline rescored file."""
    with open(data_path) as f:
        data = json.load(f)

    rollouts = []
    for r in data["results"]:
        expected = r["expected"]
        level = r["level"]
        problem = r["problem"]
        for sample in r["pass8"]:
            rollouts.append({
                "response": sample["response"],
                "predicted": sample["predicted"],  # already extracted by baseline eval
                "expected": expected,
                "level": level,
                "problem": problem,
            })

    return rollouts


def run_check(data_path: str, n_incorrect: int, seed: int) -> bool:
    random.seed(seed)

    print(f"Loading rollouts from: {data_path}")
    rollouts = load_rollouts(data_path)
    print(f"  Total rollouts: {len(rollouts)}")
    print(f"  math-verify available: {MATH_VERIFY_AVAILABLE} (grpo_train)")
    print(f"  Reference scorer: {'math-verify' if REF_AVAILABLE else 'regex fallback'}")
    print()

    # -----------------------------------------------------------------------
    # Step 1: Compute ground-truth labels using reference scorer
    # -----------------------------------------------------------------------
    print("Computing ground-truth labels (reference scorer)...")
    for r in rollouts:
        r["ref_correct"] = reference_correct(r["predicted"], r["expected"])

    correct_rollouts = [r for r in rollouts if r["ref_correct"]]
    incorrect_rollouts = [r for r in rollouts if not r["ref_correct"]]

    print(f"  Ground-truth correct:   {len(correct_rollouts)} / {len(rollouts)}")
    print(f"  Ground-truth incorrect: {len(incorrect_rollouts)} / {len(rollouts)}")

    # Sample incorrect rollouts stratified by level
    by_level = defaultdict(list)
    for r in incorrect_rollouts:
        by_level[r["level"]].append(r)

    n_levels = len(by_level)
    per_level = max(1, n_incorrect // n_levels) if n_levels else n_incorrect
    sampled_incorrect = []
    for level_rollouts in by_level.values():
        sampled_incorrect.extend(random.sample(level_rollouts, min(per_level, len(level_rollouts))))

    # Trim or top up to exactly n_incorrect
    random.shuffle(sampled_incorrect)
    sampled_incorrect = sampled_incorrect[:n_incorrect]

    print(f"  Sampled incorrect for check: {len(sampled_incorrect)}")
    print()

    # -----------------------------------------------------------------------
    # Step 2: Run GRPO reward function on correct rollouts
    # -----------------------------------------------------------------------
    print(f"Testing reward function on {len(correct_rollouts)} correct rollouts...")
    correct_mismatches = []

    for i in range(0, len(correct_rollouts), 64):
        batch = correct_rollouts[i : i + 64]
        rewards = correctness_reward(
            completions=[r["response"] for r in batch],
            answer=[r["expected"] for r in batch],
        )
        for r, reward in zip(batch, rewards):
            if reward != 1.0:
                r["reward"] = reward
                r["reward_predicted"] = extract_boxed(r["response"])
                correct_mismatches.append(r)

    # -----------------------------------------------------------------------
    # Step 3: Run GRPO reward function on sampled incorrect rollouts
    # -----------------------------------------------------------------------
    print(f"Testing reward function on {len(sampled_incorrect)} incorrect rollouts...")
    incorrect_mismatches = []

    for i in range(0, len(sampled_incorrect), 64):
        batch = sampled_incorrect[i : i + 64]
        rewards = correctness_reward(
            completions=[r["response"] for r in batch],
            answer=[r["expected"] for r in batch],
        )
        for r, reward in zip(batch, rewards):
            if reward != 0.0:
                r["reward"] = reward
                r["reward_predicted"] = extract_boxed(r["response"])
                incorrect_mismatches.append(r)

    # -----------------------------------------------------------------------
    # Step 4: Report
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("REWARD PARITY CHECK RESULTS")
    print("=" * 70)
    print(f"  Correct rollouts tested:  {len(correct_rollouts)}")
    print(f"  Reward=1 on correct:      {len(correct_rollouts) - len(correct_mismatches)} / {len(correct_rollouts)}")
    print(f"  Incorrect rollouts tested:{len(sampled_incorrect)}")
    print(f"  Reward=0 on incorrect:    {len(sampled_incorrect) - len(incorrect_mismatches)} / {len(sampled_incorrect)}")
    print()

    passed = True

    if correct_mismatches:
        passed = False
        print(f"FAIL: {len(correct_mismatches)} correct rollouts returned reward != 1.0")
        print(f"      Reward function is STRICTER than reference scorer.")
        print(f"      Likely causes: different extract_boxed regex, different normalization,")
        print(f"      stricter equality check, missing fallback parsing.")
        print()
        print("First 5 failing correct rollouts:")
        for i, r in enumerate(correct_mismatches[:5]):
            print(f"\n  [{i+1}] Level {r['level']}")
            print(f"       Expected:          {repr(r['expected'][:80])}")
            print(f"       Baseline predicted:{repr(r['predicted'][:80])}")
            print(f"       Reward extracted:  {repr(r['reward_predicted'][:80])}")
            print(f"       Reward returned:   {r['reward']}")
            print(f"       Response (first 200 chars): {repr(r['response'][:200])}")
    else:
        print(f"OK: All {len(correct_rollouts)} correct rollouts returned reward=1.0")

    if incorrect_mismatches:
        passed = False
        print(f"\nFAIL: {len(incorrect_mismatches)} incorrect rollouts returned reward != 0.0")
        print(f"      Reward function is LOOSER than reference scorer (false positives).")
        print(f"      Likely causes: substring matching, accepting any boxed expression,")
        print(f"      ignoring sign/units.")
        print()
        print("First 5 failing incorrect rollouts:")
        for i, r in enumerate(incorrect_mismatches[:5]):
            print(f"\n  [{i+1}] Level {r['level']}")
            print(f"       Expected:          {repr(r['expected'][:80])}")
            print(f"       Baseline predicted:{repr(r['predicted'][:80])}")
            print(f"       Reward extracted:  {repr(r['reward_predicted'][:80])}")
            print(f"       Reward returned:   {r['reward']}")
            print(f"       Response (first 200 chars): {repr(r['response'][:200])}")
    else:
        print(f"OK: All {len(sampled_incorrect)} incorrect rollouts returned reward=0.0")

    print()
    if passed:
        print("PASS — Reward function is in parity with reference scorer.")
        print("       Safe to launch GRPO.")
    else:
        print("FAIL — Fix reward function before launching GRPO.")

    return passed


def main():
    parser = argparse.ArgumentParser(description="Check GRPO reward function parity with baseline eval scorer")
    parser.add_argument(
        "--data",
        type=str,
        default="outputs/baseline_math500_mv_rescored.json",
        help="Path to baseline_math500_mv_rescored.json",
    )
    parser.add_argument(
        "--n_incorrect",
        type=int,
        default=100,
        help="Number of incorrect rollouts to sample for the false-positive check",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"ERROR: Data file not found: {args.data}")
        print("Run from the repo root: cd /workspace/qwen3-math-rlvr")
        sys.exit(1)

    passed = run_check(args.data, args.n_incorrect, args.seed)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
