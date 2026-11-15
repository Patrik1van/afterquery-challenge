# CERN TT20 Beam Steering Benchmark — Harbor Evaluation Summary

Evaluated with Harbor framework on Modal cloud across 5 synthetic lattice seeds.

| Agent | Task s000 | Task s001 | Task s002 | Task s003 | Task s004 | Overall (Mean ± Std) | Solved % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **oracle** (Honest Bayes, r=0.2) | 0.392 | 0.394 | 0.657 | 0.452 | 0.624 | **0.504 ± 0.114** | 0.0% |
| **opencode** (Gemini 3.8 Flash)* | 0.133 ± 0.016 | 0.198 ± 0.016 | 0.310 ± 0.037 | 0.173 ± 0.021 | 0.407 ± 0.011 | **0.244 ± 0.103** | 0.0% |
| **stop-early** | 0.001 | 0.026 | 0.000 | 0.000 | 0.000 | **0.005 ± 0.010** | 0.0% |
| **nop** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **0.000 ± 0.000** | 0.0% |

*\*Note: Earlier Gemini pilot runs scored a mean of 0.259 (~0.26). Each trial is now a fresh random puzzle from the same medium randomization distribution.*

---

## Local Honest Benchmark Comparison (Seeds 0–9)

Evaluated locally with strictly decoupled agent RNG seed (`seed + 1_000_003`), 60-shot budget, and medium domain randomization:

| Seed | Bayes (roughening = 0.2) | Bayes (roughening = 0.5) | Threading (`agents/threading.py`) | Gemini (`opencode`) |
| :---: | :---: | :---: | :---: | :---: |
| **0** | 0.4308 | 0.1586 | 0.1813 | — |
| **1** | 0.6273 | 0.1809 | 0.4790 | — |
| **2** | 0.3851 | 0.0007 | 0.3954 | — |
| **3** | 0.7439 | 0.1585 | 0.3699 | — |
| **4** | 0.6460 | 0.3916 | 0.4800 | — |
| **5** | 0.3551 | 0.1602 | 0.3978 | — |
| **6** | 0.3891 | 0.3849 | 0.0272 | — |
| **7** | 0.3944 | 0.2040 | 0.2656 | — |
| **8** | 0.3611 | 0.0501 | 0.7501 | — |
| **9** | 0.3539 | 0.2716 | 0.5700 | — |
| **Mean** | **0.4687** | **0.1961** | **0.3916** | **0.259 (~0.26)** |

---

## Reward Integrity & In-Memory Anti-Exploit Salt

### 1. Seed Leak Discovery & Resolution
In earlier development versions of the benchmark, the reference agent seed was set equal to the environment seed (`seed = env_seed`). Because the simulator sampled initial hidden offsets $\Delta\theta \sim U[-\theta_{\max}, \theta_{\max}]^{22}$ using the same NumPy RNG generator, the first row of particles in `BayesAgent` accidentally duplicated the exact ground truth offsets. When coupled with an artificially small roughening coefficient of `0.02`, the particles clustered tightly around the true offset without diffusing, achieving inflated scores of ~0.70.

When the agent seed was made independent (`env_seed + 1_000_003`), roughening `0.02` caused severe particle depletion/collapse after 1–2 shots, dropping the score to ~0.05.

### 2. Honest Reference & Operator Threading Baseline
- **Honest Bayes Reference**: With 100k particles and the physics report's roughening tuned to `0.2`, the particle filter maintains genuine particle diversity and robustness to optical model mismatch without any cheating, scoring a **mean reward of 0.4687 locally across seeds 0–9** and **0.504 on Modal cloud across tasks s000–s004**, cleanly and honestly beating Gemini (**0.24–0.26**).
- **Classic Operator Threading Baseline**: We implemented `agents/threading.py` following standard accelerator operator procedures: scanning correctors along the line in both planes simultaneously until beam level indicates presence on the foils, centering via inverse foil response $x = -\sigma \Phi^{-1}(r)$ and nominal response matrix $R$, and concluding with regularized damped least-squares refinement. This operator method achieves a **mean reward of 0.3916**, also clearly outperforming Gemini.

### 3. Closing the Seed Exploit: In-Memory Startup Salt
Because the benchmark repository is open source and task directory names reflect the nominal seed, a participant could theoretically deduce hidden offsets if the simulator relied solely on the nominal seed.

To permanently close this exploit:
1. At container startup, the sidecar server (`server.py`) generates a cryptographically random 128-bit secret salt (`salt = secrets.token_hex(16)`).
2. The salt is **kept strictly in process memory** and is never written to disk, Dockerfiles, environment variables, logs, or responses.
3. The simulator initializes its RNG from `hash(seed, salt)` via SHA-256 (`effective_seed = int.from_bytes(sha256(f"{seed}:{salt}")[:8], "big") % (2**31 - 1)`).
4. Each trial is thus a **fresh random puzzle from the same distribution**, completely un-guessable by inspecting task files.
5. All evaluation scoring is computed server-side via token-protected `/score` called exclusively by the verifier container after episode completion.

