#!/usr/bin/env python3
"""Shared prompt format for MATH few-shot evaluation and GRPO training.

Both math500_eval.py and grpo_train.py import from here so the prompt format
stays in one place. The base model is a text-completion model — this raw
few-shot string is the format it was baselined against (24.55% inferred pass@1).
"""

FEW_SHOT = """\
Problem: What is the value of $2^3 - 3 \\cdot 2 + 1$?
Solution: We compute $2^3 - 3 \\cdot 2 + 1 = 8 - 6 + 1 = 3$. The answer is $\\boxed{3}$.

Problem: If $x + y = 10$ and $x - y = 4$, what is $xy$?
Solution: Adding the equations gives $2x = 14$, so $x = 7$ and $y = 3$. Thus $xy = 7 \\cdot 3 = 21$. The answer is $\\boxed{21}$.

Problem: A right triangle has legs of length 5 and 12. What is the hypotenuse?
Solution: By the Pythagorean theorem, hypotenuse $= \\sqrt{5^2 + 12^2} = \\sqrt{25 + 144} = \\sqrt{169} = 13$. The answer is $\\boxed{13}$.

Problem: How many ways can 3 books be arranged on a shelf from 5 distinct books?
Solution: This is a permutation: $P(5,3) = 5 \\cdot 4 \\cdot 3 = 60$. The answer is $\\boxed{60}$.

"""


def create_prompt(problem: str) -> str:
    return f"{FEW_SHOT}Problem: {problem}\nSolution:"


# Stop strings for generation — shared between eval and GRPO reward truncation.
# After a solution the base model may continue the few-shot pattern with
# "\n\nProblem: ..." which would put a spurious \boxed{} after the real answer.
# math500_eval.py passes these to model.generate(); grpo_train.py's reward
# function truncates completions at the first match before extracting \boxed{}.
STOP_STRINGS = ["\n\nProblem:"]
