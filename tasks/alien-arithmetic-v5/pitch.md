# Alien Arithmetic: can a frontier model run an experiment?

> **One line:** Science means testing hypotheses on a budget. We built a small lab where a frontier model has to find out how nine unknown operators work. It keeps believing the first explanation that fits, runs tests that cannot disprove it, and then acts confidently on the wrong picture. A reference solver scores **1.00**. Gemini 4 Hydrogen scores **0.41**.

---

## 1. Motivation: models are bad at not knowing

Most benchmarks hand the model all the information and ask it to reason. Real research doesn't work that way. A scientist starts with a system they don't understand and a limited budget of experiments, then has to:

1. **Form hypotheses**: write down every explanation that fits what has been seen, not just the first one.
2. **Design experiments that tell them apart**: pick the input where the competing explanations predict *different* results.
3. **Spend a scarce budget well**: every experiment costs something, and a wasted one cannot be bought back.
4. **Know when they don't know**: commit to a conclusion only as strongly as the evidence supports, and say "no answer" when that is the truth.

This loop is how drug screening, materials discovery, debugging and lab science work. As agents are asked to run experiments for us, it matters whether they can do it.

**Our claim:** frontier models can follow this loop in words but not in practice. When an observation fits a familiar explanation, they accept it and move on. They rarely ask "what else would produce this number?", and when they do run a check, it is often one that can't come out differently.

## 2. What we built

A **two-phase lab** in which the agent has to discover hidden rules before it can use them.

```
 Phase 1: EXPERIMENT                         Phase 2: APPLY
 ┌─────────────────────────────────┐          ┌──────────────────────────────────┐
 │ 9 unknown operators ⊕ ⊗ ◇ ▽ ⊘   │  commit  │ Lab locks. 8 puzzles revealed:   │
 │ ⊛ ⊞ ⊠ △                         │  model   │   5 ? 3 ? 9 ? 9 = 154            │
 │ evaluate "900 ⊞ 13"  → 145       │ ───────▶ │ pick operators + brackets, or    │
 │ budget: 13 points, 1 per operator│ (final)  │ answer "none" if it's impossible │
 └─────────────────────────────────┘          └──────────────────────────────────┘
```

- **Hidden rules behind a sidecar.** The operators are computed by a separate `lab` container that the agent can reach only over HTTP. It can probe the rules, but it can never read them.
- **A budget enforced by the server.** 13 points, one per operator used in a query. Retrying, racing parallel requests or editing the client can't get more. Combining several probes in one query saves nothing.
- **Commit before you see the puzzles.** The agent has to write down its model of every operator before any puzzle is shown. It can't use the puzzles to work backwards to the rules.
- **Some puzzles have no solution.** The correct answer to those is `"none"`, and the agent isn't told how many there are. So it has to *trust* its model, not just fit it to the puzzles.
- **A separate verifier.** The true rules exist only in the verifier image. It brute-forces each puzzle under the true rules and grades the answers. There is no answer key that could leak.

### Why every operator is a trap

Each operator looks like a familiar operation under the most natural probe. The difference shows only if you *design* a test for it:

| Operator | True rule | What a small, lazy probe suggests | What actually reveals it |
|---|---|---|---|
| `⊞` | `(a + b) mod 256` | `a + b` | a sum ≥ 256 |
| `▽` | `abs(a − b)` | `a − b` | a < b |
| `△` | `a − b`, or 0 if a < b | `a − b` | a < b |
| `◇` | `max(a, b)` | `a` (first operand) | a < b |
| `⊠` | `a + b + n` (n = number of operands) | `a + b + 2` | a query with 3–4 operands |
| `⊘`, `⊛` | `2a − b`, `a + b²` | swapped argument order | different a and b |
| `⊕`, `⊗` | `a + b + 1`, `a·b − 1` | `a + b`, `a·b` | any careful look |

Every puzzle has 4 operands, so `⊠` adds 4 in every puzzle, not 2. A careful plan of 13 points can catch all of these. There is no spare point for a wasted probe.

## 3. Results

**Setup:** Gemini 4 Hydrogen (via opencode), 5 trials each on the fixed puzzle set and on generated puzzle sets (seeds 1–4), 25 trials in total, 60-minute limit. The **reference** is our scripted experimenter: it lists the candidate rules, picks the probe that splits them best, and commits only when exactly one rule per operator is left.

| Puzzle set | Reference | Hydrogen (5 trials) | Hydrogen mean |
|---|---|---|---|
| Fixed | 1.00 | 0.69 · 0.69 · 0.69 · 0.00 · 0.69 | 0.55 |
| Seed 1 | 1.00 | 0.44 · 0.56 · 0.25 · 0.44 · 0.38 | 0.41 |
| Seed 2 | 1.00 | 0.56 · 0.56 · 0.25 · 0.25 · 0.25 | 0.38 |
| Seed 3 | 1.00 | 0.31 · 0.31 · 0.31 · 0.13 · 0.25 | 0.26 |
| Seed 4 | 1.00 | 0.13 · 0.69 · 0.69 · 0.38 · 0.38 | 0.45 |
| **All** | **1.00** | **25 of 25 trials below the reference, best 0.69** | **0.41 (sd 0.20)** |

- **The gap is consistent:** 25 of 25 trials score below the reference and none reaches 1.0, on every puzzle set.
- **The puzzles aren't the bottleneck:** Hydrogen wrote a Python brute-forcer for the puzzles in 24 of 25 trials. Solving the puzzles is mechanical. *Knowing the rules* is the hard part.
- **Time isn't the bottleneck:** trials take 7.5 minutes on average (the longest 14.5) and commit the model after a median of about 3.7 minutes, against a 60-minute limit. No trial timed out. The model isn't running out of time. It stops investigating too early.
- **Its picture of the operators is wrong:** committed models get 5–8 of 9 operators right; none got all 9.

## 4. Where exactly frontier models fail

Each failure mode below shows up in the query logs and transcripts, and costs points through specific puzzles.

### Failure 1: never probing where the hypotheses differ
`⊞` was tested with a sum ≥ 256 in **0 of 25 trials**. The model probed with small pairs such as `(13, 5)` and `(19, 5)`, saw `14 ⊞ 5 = 19`, and concluded "the same as 14 + 5" (transcript). Result: `⊞` wrong in **24 of 25** committed models. Operands up to 1000 were allowed and documented; the model just never pushed the input range.

### Failure 2: noticing the ambiguity, then resolving it by guessing
In 2-operand probes `a + b + 1` and `a + b + i` give the same value, because i = 1 there. The model *did* see this:
> "`25` holds multiple possibilities: … `25 = a + b + 1` or `25 = a + b + i` or `25 = b ** 2` or `25 = a + 6` or `25 = a + b + n − 1`!"

It then **almost never ran the multi-operand test** that separates them (1 of 25 trials tested `⊕` with 3 or more operands). In 16 of 25 trials it committed `⊕ = a + b + i`. It listed the hypotheses correctly but spent nothing to discriminate between them.

### Failure 3: tests that can't come out differently
When the model did test `▽` with a < b, it often nested the test inside `△`:
```
(5 ▽ 19) △ 1000 = 0        4 ▽ 23 △ 1000 = 0        10 ▽ 50 △ 300 = 0
```
`△` returns 0 whenever its left side is smaller, so these results are 0 whether `▽` is `a − b` or `abs(a − b)`. The model counted it as a check anyway. Result: `▽` wrong in 16 of 25 trials. Some trials tested *only* a < b (`4 ▽ 23 = 19`) and committed `b − a`, the mirror-image mistake. `△` was wrong in 11 of 25 trials for the same reasons.

### Failure 4: lucky guesses that look like knowledge
`◇` was tested with a < b in only 4 of 25 trials, but committed correctly as `max(a, b)` in 22. The final model is right; it simply wasn't tested. Our diagnostics record the evidence separately from the outcome, so this luck is visible rather than hidden.

### Failure 5: confident action on a wrong model
- **34 false "none"s**: the model declared puzzles impossible that have a solution, because its model of the operators was wrong.
- **30 false solutions**: it "solved" puzzles that have no solution, by applying its wrong rules.
- **10 of 25 trials** submitted answers that don't follow from *their own* committed model. After seeing the puzzles, they quietly changed their beliefs instead of trusting their experiments.

### Failure 6: losing track of the resource
One trial believed it was down to its "last budget point" when it still had 4 of 13. It planned one final combined query and then **ended its turn without committing a model or writing an answer**. It scored 0 after 1.3 minutes.

**The common pattern:** the model is good at *listing* hypotheses and at *mechanical* search. It is bad at the step in between: choosing the experiment whose result would change its mind.

## 5. Why the score can be trusted

| Strategy | Reward |
|---|---|
| Do nothing | **0.00** |
| Stop before committing a model (seen live: the trial in Failure 6) | **0.00** |
| Answer `"none"` everywhere | **0.00** (a wrong "none" costs half a point) |
| Commit the "obvious" model without experimenting | **0.00** |
| Frontier model (Hydrogen) | **0.41** |
| Careful experimenter (reference) | **1.00** |

Exploit hardening:
- **Hidden state:** the rules never enter the agent's container; puzzles appear only after the model is locked.
- **Budget:** enforced by a single-threaded server, so it can't be raced. Queries the budget can't cover are refused before evaluation. Overflow errors are now *charged*, so an error can't be used as free information.
- **No working backwards:** unsolvable puzzles are generated to look solvable under at least two plausible wrong beliefs, so "the reading that makes the most puzzles solvable" points *away* from the truth.
- **Grading:** any correct expression is accepted, not only the reference solutions. Malformed answers score 0, and a verifier crash scores 0.
- **Generated puzzle sets:** `ALIEN_SEED` draws new puzzles, so results don't depend on one hand-picked set.

## 6. Demo script (10 minutes)

1. **(1 min) The hook.** Run `evaluate "14 ⊞ 5"` → 19. Ask the room: "What is ⊞?" Everyone says plus. Run `evaluate "900 ⊞ 13"` → 145.
2. **(2 min) The environment.** The two-phase diagram, the budget, the commit-then-lock rule, unsolvable puzzles.
3. **(3 min) The gap.** The results table. The reference's 13-point plan next to a Hydrogen query log (Failure 3's `(5 ▽ 19) △ 1000 = 0` is the best single slide).
4. **(3 min) The failure modes.** Failures 1–3 with the transcript quote from Failure 2: the model *lists* `a + b + i` and still never tests it.
5. **(1 min) Integrity.** The baseline table.

Reproduce:
```bash
uv run harbor run -p tasks/alien-arithmetic-v5 -a oracle -e docker            # reference → 1.0
uv run harbor run -p tasks/alien-arithmetic-v5 -a nop -e docker               # do nothing → 0.0
uv run harbor run -p tasks/alien-arithmetic-v5 -a opencode -m google/gemini-4-hydrogen -e docker -k 5 -n 5 \
  --ae GEMINI_API_KEY="$GEMINI_API_KEY" --ae GOOGLE_GENERATIVE_AI_API_KEY="$GEMINI_API_KEY"
ALIEN_SEED=1 uv run harbor run ...                                             # generated puzzle set
```
Per-trial diagnostics are in `jobs/<run>/<trial>/verifier/details.json` (which operators were wrong, false "none"s, answers inconsistent with the model).

## 7. Questions to be ready for

- **"Isn't the reference just a script that knows the answers?"** It knows a *list of candidate rules*, not which one is true, and it identifies each operator by experiment. Be upfront that its candidate list contains the true rules. The defensible claim is: "with a reasonable list of candidates and good experiment design, 13 points is enough." Better still, show a widened candidate list.
- **"Is `a + b + i` vs `a + b + 1` fair?"** It is the honest weak spot. Both fit every 2-operand probe, and the 13-point budget has no spare point to also test `⊕` and `⊗` for context-dependence. The reference never considers `a + b + i`. Answer: the simpler rule (a constant) is the right default when nothing suggests otherwise, and the model *noticed* the ambiguity and chose the more complex reading without testing it. If judges push, the fix is 1–2 extra budget points or dropping `i` and `d` from the documented variables. Decide this before judging.
- **"Is the budget too tight?"** The reference shows 13 points is enough, and Hydrogen's failures are wasted or uninformative probes, not a lack of points: it spends all 13 in 24 of 25 trials but uses them on inputs that can't discriminate.
- **"Can you train on this?"** `ALIEN_SEED` generates unlimited puzzle sets. Today the seeds change only the puzzles, not the operators. To show learning rather than memorisation, randomise the operators per seed and evaluate on held-out seeds (planned).
