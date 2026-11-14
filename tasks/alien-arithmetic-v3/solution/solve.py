"""Reference solver (the 'oracle').

It knows nothing about the hidden semantics. It works like a careful
experimenter on a budget: two probes per operator would cost 12 queries,
more than the 8 points allowed (every probe here has one operator, so costs 1),
so it probes every operator once on (100, 7),
a pair where almost every candidate gives a distinct value, and re-probes
with the operands swapped only when the first answer is still ambiguous.
It then brute-forces every puzzle over all operator choices and all five
ways to parenthesize four operands, and answers "none" when nothing reaches
the target.

Uses all 8 budget points (6 first probes + 2 re-probes for ◇ and ▽). If an operator
matches no candidate, the solver reports it and refuses to answer instead of
guessing, because a "none" built on incomplete semantics would be a guess.
"""

import itertools
import json
import os
import re
import subprocess
from pathlib import Path

SYMBOLS = ["⊕", "⊗", "◇", "▽", "⊘", "⊛"]
FIRST_PROBE, SECOND_PROBE = (100, 7), (7, 100)

CANDIDATES = {
    "a+b": lambda a, b: a + b,
    "a+b+1": lambda a, b: a + b + 1,
    "a+b-1": lambda a, b: a + b - 1,
    "a*b": lambda a, b: a * b,
    "a*b+1": lambda a, b: a * b + 1,
    "a*b-1": lambda a, b: a * b - 1,
    "a-b": lambda a, b: a - b,
    "b-a": lambda a, b: b - a,
    "|a-b|": lambda a, b: abs(a - b),
    "max": lambda a, b: max(a, b),
    "min": lambda a, b: min(a, b),
    "a": lambda a, b: a,
    "b": lambda a, b: b,
    "2a+b": lambda a, b: 2 * a + b,
    "a+2b": lambda a, b: a + 2 * b,
    "2a-b": lambda a, b: 2 * a - b,
    "2b-a": lambda a, b: 2 * b - a,
    "a+3b": lambda a, b: a + 3 * b,
    "3a+b": lambda a, b: 3 * a + b,
    "a+b^2": lambda a, b: a + b * b,
    "b+a^2": lambda a, b: b + a * a,
}

PUZZLES = {
    "P1": ([5, 3, 9, 9], 154),
    "P2": ([3, 4, 6, 8], 316),
    "P3": ([4, 3, 5, 4], 125),
    "P4": ([2, 6, 3, 8], 158),
    "P5": ([2, 3, 9, 3], 248),
    "P6": ([5, 5, 2, 5], 79),
    "P7": ([5, 5, 7, 7], 243),
    "P8": ([6, 2, 7, 9], 291),
}

# The five ways to parenthesize n0 ? n1 ? n2 ? n3 (shape 0 is left to right).
SHAPES = [
    ("{0} {a} {1} {b} {2} {c} {3}", lambda f, n: f(2, f(1, f(0, n[0], n[1]), n[2]), n[3])),
    ("({0} {a} ({1} {b} {2})) {c} {3}", lambda f, n: f(2, f(0, n[0], f(1, n[1], n[2])), n[3])),
    ("({0} {a} {1}) {b} ({2} {c} {3})", lambda f, n: f(1, f(0, n[0], n[1]), f(2, n[2], n[3]))),
    ("{0} {a} (({1} {b} {2}) {c} {3})", lambda f, n: f(0, n[0], f(2, f(1, n[1], n[2]), n[3]))),
    ("{0} {a} ({1} {b} ({2} {c} {3}))", lambda f, n: f(0, n[0], f(1, n[1], f(2, n[2], n[3])))),
]


def query(expr: str) -> int:
    out = subprocess.run(["evaluate", expr], capture_output=True, text=True, check=True).stdout
    return int(re.search(r"=\s*(-?\d+)", out).group(1))


def matching(probe, value, names):
    return [n for n in names if CANDIDATES[n](*probe) == value]


def identify():
    semantics = {}
    for sym in SYMBOLS:
        matches = matching(FIRST_PROBE, query("{} {} {}".format(FIRST_PROBE[0], sym, FIRST_PROBE[1])), CANDIDATES)
        if len(matches) > 1:
            value = query("{} {} {}".format(SECOND_PROBE[0], sym, SECOND_PROBE[1]))
            matches = matching(SECOND_PROBE, value, matches)
        print(f"{sym}: {matches or 'no candidate'}")
        if len(matches) == 1:
            semantics[sym] = CANDIDATES[matches[0]]
    return semantics


def solve(semantics):
    if set(semantics) != set(SYMBOLS):
        print("semantics incomplete; not answering")
        return {}
    answer = {}
    for pid, (nums, target) in PUZZLES.items():
        answer[pid] = "none"
        for ops in itertools.product(semantics, repeat=3):
            f = lambda i, a, b: semantics[ops[i]](a, b)
            hit = next((fmt for fmt, shape in SHAPES if shape(f, nums) == target), None)
            if hit:
                answer[pid] = hit.format(*nums, a=ops[0], b=ops[1], c=ops[2])
                break
    return answer


if __name__ == "__main__":
    sem = identify()
    ans = solve(sem)
    Path(os.environ.get("ANSWER_PATH", "/app/answer.json")).write_text(json.dumps(ans, ensure_ascii=False))
    print(json.dumps(ans, ensure_ascii=False))
