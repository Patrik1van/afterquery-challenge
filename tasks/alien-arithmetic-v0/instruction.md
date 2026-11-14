# Alien arithmetic

You are in a lab that uses four binary operators on integers:

    ⊕   ⊗   ◇   ▽

Their meaning is unknown to you. Do not assume they behave like any familiar
operation; find out what they do by experiment.

## Your tool

The `evaluate` command asks the lab to compute an expression:

```bash
evaluate "2 ⊕ 3"
evaluate "4 ◇ 1 ▽ 6"
evaluate --budget        # free: shows how many queries you have left
```

- Expressions are evaluated **strictly left to right**: `a ? b ? c` means `(a ? b) ? c`. There are no parentheses.
- Operands are integers from 0 to 1000, at most 4 operands per expression.
- **You have a budget of 10 queries in total.** Every successfully evaluated expression costs one query; malformed expressions return an error and cost nothing. When the budget is used up, the lab refuses further queries.

## The puzzles

In each puzzle, replace every `?` with one of the four operators so that the
equation holds. The numbers stay in the given order, evaluation is left to
right, and operators may repeat. Each puzzle has **exactly one** solution.

```
P1:  5 ? 3 ? 8 ? 2  = 74
P2:  2 ? 3 ? 4 ? 8  = 28
P3:  3 ? 3 ? 6 ? 9  = 14
P4:  7 ? 8 ? 3 ? 5  = 16
P5:  3 ? 5 ? 9 ? 5  = 39
P6:  5 ? 6 ? 8 ? 7  = 22
```

## Your answer

Write your answer to `/app/answer.json`: one string of three operator symbols per
puzzle, in order. For example:

```json
{"P1": "⊕⊕⊕", "P2": "⊗◇▽", "P3": "...", "P4": "...", "P5": "...", "P6": "..."}
```

You are scored on how many puzzles you solve correctly. Answer every puzzle;
a missing answer counts as wrong.
