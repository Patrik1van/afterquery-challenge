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
- **You have a budget of 8 queries in total.** Every successfully evaluated expression costs one query; malformed expressions return an error and cost nothing. When the budget is used up, the lab refuses further queries.

## The puzzles

In each puzzle, replace every `?` with one of the six operators and, if you
need to, add parentheses, so that the expression equals the target. The
numbers stay in the given order, operators may repeat, and an expression
without parentheses is evaluated left to right.

**Some puzzles have no solution.** For those, the correct answer is `"none"`.

```
P1:  5 ? 3 ? 9 ? 9  = 154
P2:  3 ? 4 ? 6 ? 8  = 316
P3:  4 ? 3 ? 5 ? 4  = 125
P4:  2 ? 6 ? 3 ? 8  = 158
P5:  2 ? 3 ? 9 ? 3  = 248
P6:  5 ? 5 ? 2 ? 5  = 79
P7:  5 ? 5 ? 7 ? 7  = 243
P8:  6 ? 2 ? 7 ? 9  = 291
```

## Your answer

Write your answer to `/app/answer.json`. For each puzzle, give either one
complete expression (the puzzle's numbers in the given order, exactly three
operators) or `"none"` if the puzzle has no solution. The format looks like
this (the values are only illustrations, not hints):

```json
{"P1": "5 ⊕ 3 ⊕ 9 ⊕ 9", "P2": "(3 ⊗ 4) ◇ (6 ▽ 8)", "P3": "...", "P4": "...", "P5": "...", "P6": "...", "P7": "...", "P8": "none"}
```

An expression is correct if the lab's operators make it equal to the target;
`"none"` is correct only if no expression reaches the target. You are scored
on how many puzzles you answer correctly. Answer every puzzle; a missing
answer counts as wrong.
