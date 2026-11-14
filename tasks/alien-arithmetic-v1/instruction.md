# Alien arithmetic

You are in a lab that uses six binary operators on integers:

    ⊕   ⊗   ◇   ▽   ⊘   ⊛

Their meaning is unknown to you. Do not assume they behave like any familiar
operation; find out what they do by experiment.

## Your tool

The `evaluate` command asks the lab to compute an expression:

```bash
evaluate "2 ⊕ 3"
evaluate "4 ◇ 1 ▽ 6"
evaluate "4 ◇ (1 ▽ 6)"
evaluate --budget        # free: shows how many queries you have left
```

- Without parentheses, expressions are evaluated **strictly left to right**: `a ? b ? c` means `(a ? b) ? c`.
- Parentheses group sub-expressions and may be nested: `a ? (b ? c)`.
- Operands are integers from 0 to 1000, at most 4 operands per expression.
- **You have a budget of 10 queries in total.** Every successfully evaluated expression costs one query; malformed expressions return an error and cost nothing. When the budget is used up, the lab refuses further queries.

## The puzzles

In each puzzle, replace every `?` with one of the six operators and, if you
need to, add parentheses, so that the expression equals the target. The
numbers stay in the given order, operators may repeat, and an expression
without parentheses is evaluated left to right. Every puzzle has at least
one solution.

```
P1:  6 ? 8 ? 3 ? 7  = 141
P2:  3 ? 9 ? 2 ? 5  = 107
P3:  2 ? 8 ? 4 ? 8  = 340
P4:  4 ? 9 ? 4 ? 6  = 205
P5:  8 ? 8 ? 7 ? 2  = 92
P6:  5 ? 4 ? 3 ? 4  = 255
```

## Your answer

Write your answer to `/app/answer.json`: one complete expression per puzzle,
using the puzzle's numbers in the given order and exactly three operators.
For example:

```json
{"P1": "6 ⊕ 8 ⊕ 3 ⊕ 7", "P2": "(3 ⊗ 9) ◇ (2 ▽ 5)", "P3": "...", "P4": "...", "P5": "...", "P6": "..."}
```

An answer is correct if the lab's operators make it equal to the target.
You are scored on how many puzzles you solve correctly. Answer every puzzle;
a missing answer counts as wrong.
