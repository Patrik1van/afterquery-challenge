# Alien arithmetic

You are in a lab that uses six binary operators on integers:

    ⊕   ⊗   ◇   ▽   ⊘   ⊛

Their meaning is unknown to you. Do not assume they behave like any familiar
operation; find out what they do by experiment.

This task has **two phases**. In phase 1 you experiment, but you do not see the
puzzles. In phase 2 you see the puzzles, but you can no longer experiment.

## Phase 1: experiment

The `evaluate` command asks the lab to compute an expression:

```bash
evaluate "2 ⊕ 3"         # costs 1
evaluate "4 ◇ 1 ▽ 6"     # costs 2
evaluate "4 ◇ (1 ▽ 6)"   # costs 2
evaluate --budget        # free: shows how much budget you have left
```

- Without parentheses, expressions are evaluated **strictly left to right**: `a ? b ? c` means `(a ? b) ? c`.
- Parentheses group sub-expressions and may be nested: `a ? (b ? c)`.
- Operands are integers from 0 to 1000, at most 4 operands per expression.
- **You have a budget of 8 points in total.** A successfully evaluated expression costs **one point per operator** in it: `a ? b` costs 1, `a ? b ? c` costs 2, `(a ? b) ? (c ? d)` costs 3. Malformed expressions return an error and cost nothing, and so does an expression that costs more than you have left. When the budget is used up, the lab refuses further queries.

## Committing your model

When you are done experimenting, write your model of every operator to
`/app/model.json`: a formula in `a` and `b` for each symbol, meaning `a ? b`.
The format looks like this (the formulas are placeholders, not hints):

```json
{"⊕": "a + b", "⊗": "a * b", "◇": "min(a, b)", "▽": "a - b", "⊘": "a", "⊛": "b"}
```

Formulas may use integers, `a`, `b`, `+ - * // % **`, `abs`, `max`, `min`,
comparisons, `and`/`or`/`not` and `x if condition else y`.

Then run:

```bash
reveal /app/model.json
```

**This is final.** The lab checks the model. If it is valid, the lab stores it,
**locks itself** (`evaluate` stops working) and shows you the puzzles, which
are also saved to `/app/puzzles.json`. If the model is invalid, it is rejected,
nothing is locked, and you can fix it and try again. Your committed model is
recorded, but it is not what you are scored on; the puzzles are.

## Phase 2: the puzzles

There are 8 puzzles, `P1` to `P8`. Each has the form `n1 ? n2 ? n3 ? n4 = target`.
Replace every `?` with one of the six operators and, if you need to, add
parentheses, so that the expression equals the target. The numbers stay in the
given order, operators may repeat, and an expression without parentheses is
evaluated left to right.

**Some puzzles have no solution.** For those, the correct answer is `"none"`.

## Your answer

Write your answer to `/app/answer.json`. For each puzzle, give either one
complete expression (the puzzle's numbers in the given order, exactly three
operators) or `"none"` if the puzzle has no solution. The format looks like
this (the values are only illustrations, not hints):

```json
{"P1": "1 ⊕ (2 ⊗ 3) ◇ 4", "P2": "none", "P3": "...", "P4": "...", "P5": "...", "P6": "...", "P7": "...", "P8": "..."}
```

An expression is correct if the lab's operators make it equal to the target;
`"none"` is correct only if no expression reaches the target. You are scored
on how many puzzles you answer correctly. Answer every puzzle; a missing
answer counts as wrong.
