# CERN TT20 Beam Steering Benchmark — Harbor Evaluation Summary

Evaluated with Harbor framework on Modal cloud across 5 synthetic lattice seeds.

| Agent | Task s000 | Task s001 | Task s002 | Task s003 | Task s004 | Overall (Mean ± Std) | Solved % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **oracle** (Bayes, r=0.2) | 0.518 | 0.390 | 0.379 | 0.485 | 0.444 | **0.443 ± 0.054** | 0.0% |
| **opencode** (Gemini 3.8 Flash)* | 0.231 ± 0.139 | 0.259 ± 0.087 | 0.299 ± 0.034 | 0.194 ± 0.034 | 0.314 ± 0.132 | **0.259 ± 0.106** | 0.0% |
| **stop-early** | 0.001 | 0.026 | 0.000 | 0.000 | 0.000 | **0.005 ± 0.010** | 0.0% |
| **nop** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **0.000 ± 0.000** | 0.0% |

*\*Note: The 15 Gemini runs (3 attempts across 5 seeds, mean 0.259) were conducted on an earlier task build with the identical machine, noise, and perturbation distribution.*

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

## Reward Integrity & Seed Exploit Fix

### 1. The Seed Leak Bug and Investigation
In earlier iterations of the benchmark, the reference agent seed was set equal to the environment seed (`seed = env_seed`). Because the simulator sampled initial hidden offsets $\Delta\theta \sim U[-\theta_{\max}, \theta_{\max}]^{22}$ using the same NumPy RNG generator, the first row of particles in `BayesAgent` accidentally duplicated the exact ground truth offsets. When coupled with an artificially small roughening coefficient of `0.02`, the particles clustered tightly around the true offset without diffusing, achieving inflated scores of ~0.70.

When the agent seed was made independent (`env_seed + 1_000_003`), roughening `0.02` caused severe particle depletion/collapse after 1–2 shots, dropping the score to ~0.05.

### 2. Resolution: Honest Bayes & Scripted Threading Reference
- **Honest Bayes Reference**: With 100k particles and the physics report's roughening tuned to `0.2`, the particle filter maintains genuine particle diversity and robustness to optical model mismatch without any cheating, scoring a **mean reward of 0.4687 locally across seeds 0–9** and **0.443 on Modal cloud across tasks s000–s004**, cleanly and honestly beating Gemini (**0.259**).
- **Classic Operator Threading Baseline**: We wrote `agents/threading.py`, implementing progressive beam threading: scanning correctors along the line in both planes simultaneously until beam level indicates presence on the foils, centering via inverse foil response $x = -\sigma \Phi^{-1}(r)$ and nominal response matrix $R$, and concluding with regularized damped least-squares refinement. This operator method achieves a **mean reward of 0.3916**, also clearly outperforming Gemini (**0.259**).

### 3. Closing the Seed Exploit via Secret Salt
Because benchmark code is public and task directories typically indicate the seed in their name, an agent could theoretically reconstruct the simulator's random stream if the seed were directly used. 

To permanently close this exploit:
1. Each generated task receives a secret 128-bit random salt (`BEAMLINE_SALT`), generated via `secrets.token_hex(16)`.
2. This salt is baked **only into the sidecar container image** (`environment/beamline/Dockerfile`).
3. The sidecar derives its effective simulator seed from `hash(seed, salt)` via SHA-256.
4. Neither the salt nor the seed is provided in `instruction.md` or the agent container environment.
5. All evaluation scoring is computed server-side via token-protected `/score` called exclusively by the verifier container after episode completion.

