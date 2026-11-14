# v2 observations: Gemini 3.8 Flash, 1800 s timeout

Job `jobs/2026-11-14__16-35-19`: opencode + `google/gemini-3.8-flash`, 5 trials, budget 8.

## Summary

- **Mean 0.175**, below the 0.25 you'd get by answering `"none"` everywhere. No timeouts.
- **One wrong belief explains almost every loss:** probing with `2 ⊕ 3 = 6`, `2 ⊗ 3 = 5` and reading them as `a·b` / `a+b` (true: `a+b+1` / `a·b−1`).
- **The unsolvable-puzzle traps fired in 5/5 trials.** A model with a wrong picture of the operators "solves" P3 or P6.
- **No trial deliberately hedged with `"none"`.** Every false "none" came from a wrong belief that made the puzzle look unsolvable.

| Trial | ⊕/⊗ probe | Reward | Wrong belief |
|---|---|---|---|
| `cfr7Yx2` | `(5, 3)` | **0.625** | `⊘ = a+b−1` |
| `mq4E9UT` | `(2, 3)` | 0.125 | ⊕/⊗ swapped |
| `v45eLAe` | `(2, 3)` | 0.125 | ⊕/⊗ swapped |
| `3KxpojU` | `(2, 3)` | 0.0 | ⊕/⊗ swapped |
| `zbidMvB` | `(2, 3)` | 0.0 | ⊕/⊗ swapped |

## Example 1: the ⊕/⊗ swap (4 of 5 trials)

`v45eLAe` queries: `2 ⊕ 3=6 | 2 ⊗ 3=5 | 10 ◇ 3=10 | 10 ▽ 3=7 | 10 ⊘ 3=17 | 10 ⊛ 3=19 | 3 ◇ 10=10 | 1 ⊘ 1=1`

In its own words: *"⊕ is indeed multiplication (\*), and ⊗ is addition (+)"*. `mq4E9UT` hard-coded a single
candidate, `cand_plus_circ = {'mul': lambda a, b: a * b}`, and never considered `a+b+1`.

It spent its 2 spare queries on `◇` and `⊘`. Neither challenged the ⊕/⊗ reading, which mattered most.

## Example 2: the traps work

All four "swapped" trials gave **the same answers** to P3, P4 and P6. Each answer is exact under the wrong
belief and wrong under the true rules:

| Puzzle | Their answer | Under ⊕=a·b, ⊗=a+b | True value | Truth |
|---|---|---|---|---|
| P3 = 125 | `4 ⊛ ((3 ⊕ 5) ▽ 4)` | 4 + (15−4)² = **125** | 29 | no solution |
| P6 = 79 | `(5 ⊛ 5) ⊛ (2 ⊗ 5)` | 30 + 7² = **79** | 111 | no solution |
| P4 = 158 | `2 ⊕ ((6 ⊛ 3) ⊛ 8)` | 2 · 79 = **158** | 82 | `(2 ⊛ (6 ⊘ 3)) ⊘ 8` |

Under the same belief, most solvable puzzles looked unsolvable. That's where the 3–5 false "none"s per trial come from.

## Example 3: the best trial still had one ambiguous probe

`cfr7Yx2` probed everything on `(5, 3)`, so ⊕ (9) and ⊗ (14) were unambiguous. It got 5/8.
But `5 ⊘ 3 = 7` fits both `2a−b` (true) and `a+b−1`. It concluded `a+b−1` and never re-probed `⊘`:

- P1 and P4 need `⊘`, so it answered `"none"` (false none).
- P3 (unsolvable): `(4 ⊘ 3) ⊗ (5 ⊛ 4)` gives 6·21−1 = **125** under its belief (false solution).

## Takeaways

1. **The task discriminates on experimental design, as intended.** Small symmetric probes like `(2, 3)`
   collide across many simple rules. The oracle's `(100, 7)` + `(7, 100)` doesn't, and solves 8/8.
2. **Gemini doesn't check readings that fit its data.** No trial spent a spare query testing its most
   consequential assumption.
3. **The reward is gameable in principle.** Answering `"none"` everywhere (0.25) beats this model (0.175).
   Nobody exploited it here, but a `false_none` penalty is the next lever if stronger or trained models start hedging.
