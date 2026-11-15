"""Deterministic generator of puzzle sets "n1 ? n2 ? n3 ? n4 = target" over the lab's operators."""

import itertools
import random


def table(semantics, key):
    """Build callables for one column of the semantics.

    Args:
        semantics: Mapping of symbol to {"rule": ..., "naive": ...} formula strings (trusted input).
        key: "rule" for the true operators or "naive" for the plausible wrong readings.

    Returns:
        Mapping of symbol to f(a, b) evaluated at n = 4, the operand count of every puzzle.
    """
    return {s: eval(f"lambda a, b: {v[key]}", {"n": 4}) for s, v in semantics.items()}


def bracketings(f, ops, nums):
    """Return the values of n1 o1 n2 o2 n3 o3 n4 for all five ways to place parentheses."""
    (A, B, C), (w, x, y, z) = (f[s] for s in ops), nums
    return [C(B(A(w, x), y), z), C(A(w, B(x, y)), z), B(A(w, x), C(y, z)), A(w, C(B(x, y), z)), A(w, B(x, C(y, z)))]


def solutions(f, nums, target):
    """Return every (operators, bracketing index) that reaches the target."""
    return {(o, i) for o in itertools.product(f, repeat=3) for i, v in enumerate(bracketings(f, o, nums)) if v == target}


def generate(semantics, seed, difficulty=3):
    """Draw a set of 8 puzzles.

    Difficulty 1 gives 8 solvable puzzles with at most 2 solutions each. Difficulty 2 gives
    7 solvable and 1 unsolvable, where the naive reading of any operator a solvable puzzle
    uses gives a wrong answer. Difficulty 3 gives 6 solvable and 2 unsolvable, and together
    the solvable puzzles catch the naive reading of every operator. Unsolvable puzzles look
    solvable under at least two naive readings.

    Args:
        semantics: Mapping of symbol to {"rule": ..., "naive": ...} formula strings.
        seed: Random seed; the same seed and difficulty always give the same puzzles.
        difficulty: 1, 2 or 3.

    Returns:
        Mapping "P1".."P8" to {"numbers": [n1, n2, n3, n4], "target": t}.
    """
    rng = random.Random(seed)
    n_unsolvable = difficulty - 1
    true, naive = table(semantics, "rule"), table(semantics, "naive")
    beliefs = {s: dict(true, **{s: naive[s]}) for s in true}
    while True:
        solvable, unsolvable, caught = [], [], set()
        for _ in range(3000):
            if len(solvable) == 8 - n_unsolvable and len(unsolvable) == n_unsolvable:
                puzzles = solvable + unsolvable
                rng.shuffle(puzzles)
                return {f"P{k}": {"numbers": n, "target": t} for k, (n, t) in enumerate(puzzles, 1)}
            nums = [rng.randint(2, 9) for _ in range(4)]
            ops = [rng.choice(list(true)) for _ in range(3)]
            target = rng.randint(10, 400) if rng.random() < 0.3 else bracketings(true, ops, nums)[rng.randrange(5)]
            if not 10 <= target <= 400:
                continue
            sols = solutions(true, nums, target)
            if not sols:
                if len(unsolvable) < n_unsolvable and sum(bool(solutions(b, nums, target)) for b in beliefs.values()) >= 2:
                    unsolvable.append((nums, target))
            elif len(sols) <= 2 and len(solvable) < 8 - n_unsolvable:
                used = {s for o, _ in sols for s in o}
                need = -(-(len(true) - len(caught)) // (8 - n_unsolvable - len(solvable))) if difficulty >= 3 else 0
                traps_hold = difficulty < 2 or all(not (solutions(beliefs[s], nums, target) & sols) for s in used)
                if len(used - caught) >= need and traps_hold:
                    solvable.append((nums, target))
                    caught |= used
