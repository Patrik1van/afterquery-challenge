"""Puzzle generator: generate(semantics, seed, difficulty) -> {"P1": {"numbers": [...], "target": t}, ...}.

Every puzzle is "n1 ? n2 ? n3 ? n4 = target" over the lab's operators (8 puzzles).
  difficulty 1: all 8 solvable, at most 2 solutions each.
  difficulty 2: 7 solvable + 1 unsolvable. A naive reading of any operator a
                solvable puzzle uses gives a wrong answer, so guessing doesn't pay.
  difficulty 3: 6 solvable + 2 unsolvable, as in 2, and together the solvable
                puzzles catch the naive reading of every operator.
Unsolvable puzzles look solvable under at least two naive readings.
Same seed and difficulty, same puzzles.
"""

import itertools
import random


def table(semantics, key):
    """symbol -> f(a, b) at n = 4, the operand count of every puzzle. Trusted: the lab's own formulas."""
    return {s: eval(f"lambda a, b: {v[key]}", {"n": 4}) for s, v in semantics.items()}


def bracketings(f, ops, nums):
    """Values of n1 o1 n2 o2 n3 o3 n4 for all 5 ways to place parentheses."""
    (A, B, C), (w, x, y, z) = (f[s] for s in ops), nums
    return [C(B(A(w, x), y), z), C(A(w, B(x, y)), z), B(A(w, x), C(y, z)), A(w, C(B(x, y), z)), A(w, B(x, C(y, z)))]


def solutions(f, nums, target):
    """Every (operators, bracketing) that reaches the target."""
    return {(o, i) for o in itertools.product(f, repeat=3) for i, v in enumerate(bracketings(f, o, nums)) if v == target}


def generate(semantics, seed, difficulty=3):
    rng = random.Random(seed)
    n_unsolvable = difficulty - 1
    true, naive = table(semantics, "rule"), table(semantics, "naive")
    beliefs = {s: dict(true, **{s: naive[s]}) for s in true}  # one naive reading at a time
    while True:  # deterministic retries until a full set is drawn
        solvable, unsolvable, caught = [], [], set()
        for _ in range(3000):
            if len(solvable) == 8 - n_unsolvable and len(unsolvable) == n_unsolvable:
                puzzles = solvable + unsolvable
                rng.shuffle(puzzles)
                return {f"P{k}": {"numbers": n, "target": t} for k, (n, t) in enumerate(puzzles, 1)}
            nums = [rng.randint(2, 9) for _ in range(4)]
            ops = [rng.choice(list(true)) for _ in range(3)]
            # mostly a target some expression reaches, sometimes a random one (candidates for "none")
            target = rng.randint(10, 400) if rng.random() < 0.3 else bracketings(true, ops, nums)[rng.randrange(5)]
            if not 10 <= target <= 400:
                continue
            sols = solutions(true, nums, target)
            if not sols:
                if len(unsolvable) < n_unsolvable and sum(bool(solutions(b, nums, target)) for b in beliefs.values()) >= 2:
                    unsolvable.append((nums, target))
            elif len(sols) <= 2 and len(solvable) < 8 - n_unsolvable:
                used = {s for o, _ in sols for s in o}
                # difficulty 3: each slot must catch enough new operators to cover all of them
                need = -(-(len(true) - len(caught)) // (8 - n_unsolvable - len(solvable))) if difficulty >= 3 else 0
                traps_hold = difficulty < 2 or all(not (solutions(beliefs[s], nums, target) & sols) for s in used)
                if len(used - caught) >= need and traps_hold:
                    solvable.append((nums, target))
                    caught |= used
