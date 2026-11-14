# Algorithms for Steering the CERN TT20 Transfer Line

Adam Štafa (supervisors: Adrián Menor de Oñate, Michael Schenk), 25 Sept 2026.
Markdown conversion of `papers/beamline_steering.pdf` for internal team use. Equation numbers
match the original. Figures are omitted — see the PDF.

## Abstract
After every annual shutdown, operators commission the TT20 transfer line manually (several
hours). The report automates this with reinforcement learning and Bayesian parameter estimation.
Both steer the beamline in simulation and handle randomized dynamics.

## 1 Introduction
TT20 delivers protons from the SPS to North Area fixed-target experiments. Steering uses
**10 horizontal and 12 vertical corrector dipoles** (each deflects the beam by an angle in its
plane) and **secondary emission monitors (SEMs), 11 horizontal and 10 vertical**, each measuring
beam intensity on two foils, which gives a sense of beam position. In an unsteered line the beam
may hit only one foil or the beampipe, so position can't be estimated reliably — this makes
classical control hard.

### 1.1.1 Beamline model

**Response matrix.** Linear, uncoupled optics (planes independent, trajectory linear in corrector
angles). With Twiss β and μ (μ in units of 2π, from MAD-X), corrector j with angle θⱼ displaces
the trajectory at element i by

$$x_i = \sum_j R_{ij}\,\theta_j,\qquad
R_{ij} = \begin{cases}\sqrt{\beta_i\beta_j}\,\sin\big(2\pi(\mu_i-\mu_j)\big), & \mu_i>\mu_j\\ 0 & \text{otherwise}\end{cases} \tag{1}$$

separately for each plane.

**Beam cross-section.** Gaussian profile at every monitor, σᵢ = √(βᵢ ε), geometric emittance
ε = εₙ/(βγ) with normalized emittance εₙ = 2 µm and βγ = 427.35. Each SEM has two foils of
length L. For a beam centred at x, intercepted fractions (Φ = standard normal CDF):

$$I_1(x) = \Phi\!\left(\tfrac{-x}{\sigma}\right) - \Phi\!\left(\tfrac{-L-x}{\sigma}\right),\qquad
I_2(x) = \Phi\!\left(\tfrac{L-x}{\sigma}\right) - \Phi\!\left(\tfrac{-x}{\sigma}\right) \tag{2}$$

Readings are corrupted by noise of magnitude ν, drawn independently per foil per step:

$$\tilde I_j = I_j + \epsilon_j,\qquad \epsilon_j\sim U[0,\nu] \tag{3}$$

Only normalized fractions and the total level (divided by I_max, the intensity of a centred beam)
are reported per SEM:

$$\frac{\tilde I_1}{\tilde I_1+\tilde I_2},\quad \frac{\tilde I_2}{\tilde I_1+\tilde I_2},\quad \frac{\tilde I_1+\tilde I_2}{I_{\max}} \tag{4}$$

**Collision detection.** Rectangular beampipe with per-element half-widths a_h, a_v from the
lattice. The beam is lost at the first element where |x_h| > a_h or |x_v| > a_v; every monitor
from there on reads background only.

### 1.1.2 Steering goal

**Symmetry** per monitor (1 = centred, hits both foils equally; 0 = hits only one foil):

$$S = \sqrt{1 - \frac{|I_1-I_2|}{I_1+I_2}} \tag{5}$$

Multiplied by achieved intensity to cover beam loss:

$$\hat S = S\,\frac{I_1+I_2}{I_{\max}} \tag{6}$$

**An episode is solved when the mean Ŝ over all monitors exceeds 0.95.**

**Standard-deviation-normalized distance** (used for evaluation / auxiliary training signal;
symmetry saturates once roughly centred and is flat when the beam misses the foils):

$$d = \sqrt{\frac1N\sum_i\left(\frac{x_i}{\sigma_i}\right)^2} \tag{7}$$

### 1.2 Limitations
Gaussian beam; rectangular aperture everywhere (some are elliptical in reality); binary loss
when the beam centre leaves the aperture (no partial scraping); uncorrelated uniform noise;
perfect steering is possible in simulation but not in reality.

## 2 Reinforcement learning (PPO) — summary
- Each step: observe monitor readings, propose a *relative* change of correctors, get reward
  $$r = \frac1N\sum_i(\hat S_i - 1) - 0.05\cdot\frac1N\sum_i\frac{|x_i|}{\sigma_i} \tag{8}$$
  (symmetry alone failed to learn: with lost beam the reward is ≈ −1 everywhere; the distance
  term uses true positions, available only in simulation).
- Actions shaped cubically a ↦ (a/2)³ and expressed in **monitor space**: displacement Δ in beam
  widths, mapped to correctors by the pseudo-inverse of the width-normalized response matrix
  $$\Delta\theta = \tilde R^{+}\Delta,\qquad \tilde R_{ij} = R_{ij}/\sigma_i \tag{9}$$
- JAX implementation, 256 parallel envs, 50-step rollouts, linear LR decay. Episodes start from
  correctors drawn U[−θmax, θmax] (beam typically lost after the first few monitors); end at
  solved, truncated at 150 steps.
- **Domain randomization**: bank of 1024 perturbed beamlines (Table 1, medium spreads).
- Disadvantages: training fragile (depends on action parameterization, reward coefficient,
  hyperparameters; high seed variance; retrain when the model changes). **History is hard to
  include** — the policy only sees the last observation; frame stacking and recurrent nets were
  hard to train and performed worse, yet an optimal policy needs to remember what it tried.

**Table 1 — randomized parameters (rms spread)**

| Parameter | low | medium | high | high + Student-t |
|---|---|---|---|---|
| quadrupole gradient | ±0.05 % | ±0.1 % | ±0.2 % | ±0.2 % |
| corrector gain | ±1 % | ±2 % | ±4 % | ±4 % |
| initial βx | ±25 % | ±50 % | ±100 % | ±100 % |
| initial βy | ±5 % | ±10 % | ±20 % | ±20 % |
| initial αx | ±0.5 | ±1.0 | ±2.0 | ±2.0 |
| initial αy | ±0.15 | ±0.3 | ±0.6 | ±0.6 |
| emittance | ±10 % | ±20 % | ±40 % | ±40 % |
| background level | ±20 % | ±40 % | ±80 % | ±80 % |
| beam profile | Gaussian | Gaussian | Gaussian | Student-t, ν=3 |

Relative errors are log-normal (spread = rms of the log), except quadrupole gradients; initial α
shifted by an absolute amount. Errors drawn independently per quadrupole, corrector and plane.

## 3 Bayesian estimation (← our reference agent)

### 3.1 Motivation
There's no real dependency between actions — the agent observes the line's response to proposed
settings — so treat it as **inference of the unknown initial offset**.

### 3.2 Formulation
Each corrector is initially missteered; offsets Δθ ∈ ℝ²² (both planes). Beam position:

$$x = R(\Delta\theta - \theta) \tag{10}$$

The agent observes eq. (4). For SEM k, ratio and level:

$$r_k = \frac{\tilde I_{1,k}}{\tilde I_{1,k}+\tilde I_{2,k}},\qquad \ell_k = \frac{\tilde I_{1,k}+\tilde I_{2,k}}{I_{\max,k}} \tag{11}$$

(the second fraction = 1 − r_k, no extra info). Infer Δθ from observations under different θ.
Response matrix, beam widths and background bound are known from the model.

### 3.3 Bayes recursion
Uniform prior

$$p(\Delta\theta) = U\big([-\theta_{\max},\theta_{\max}]^{22}\big) \tag{12}$$

update after each commanded θ_t and observation o_t:

$$p(\Delta\theta\mid o_{1:t}) \propto p(o_t\mid\Delta\theta,\theta_t)\;p(\Delta\theta\mid o_{1:t-1}) \tag{13}$$

Δθ is static, so the posterior is the prior times all likelihoods so far.

### 3.4 Particle filter
Belief represented by **10⁵ particles**. Each update: score each particle (eq. 15), normalize,
**multinomial resampling**.

**Approximate likelihood** (exact likelihood depletes particles fast → wrong convergence). For
particle Δθ: trajectory x = R(Δθ − θ_t); foil intensities with expected background added,
Ĩ_{j,k} = I_j(x_k) + ν/2; sum T̃_k = Ĩ_{1,k} + Ĩ_{2,k}. Residuals normalized by the band the
background can move each reading:

$$z^r_k = \frac{r_k - \tilde I_{1,k}/\tilde T_k}{\nu/\tilde T_k},\qquad
z^\ell_k = \frac{\ell_k - \tilde T_k/I_{\max,k}}{2\nu/I_{\max,k}} \tag{14}$$

$$\log p(o_t\mid\Delta\theta,\theta_t) = -\sum_k\Big(h(z^r_k) + h(z^\ell_k)\Big),\qquad
h(z) = \begin{cases}\tfrac12 z^2, & |z|\le1\\ |z|-\tfrac12, & |z|>1\end{cases} \tag{15}$$

(Huber loss.) The ratio band ν/T̃_k grows as the predicted beam leaves the foils, so a monitor
seeing only background switches itself off. **The ratio term is dropped at monitors where the
particle predicts the beam to be lost.**

**Roughening.** After each resampling, add jitter with the covariance Σ of the current population:

$$\Delta\theta \leftarrow \Delta\theta + 0.5\,\Sigma^{1/2}\varepsilon,\qquad \varepsilon\sim\mathcal N(0,I) \tag{16}$$

This regenerates spread only along poorly determined directions — important because TT20's
response matrix has near-null directions no measurement can resolve.

### 3.5 Sampling (which θ to command)
Posterior mean centres the beam if right but explores poorly; Thompson sampling explores well but
wastes steps once the posterior is narrow. Interpolate:

$$\theta = \overline{\Delta\theta} + \big(\alpha + \beta\,p_{\text{lost}}\big)\big(\Delta\theta^{(s)} - \overline{\Delta\theta}\big),\qquad \alpha=0.1,\ \beta=0.6 \tag{17}$$

Δθ⁽ˢ⁾ = a uniform draw from the population; p_lost = fraction of particles predicting the beam is
lost before the end of the line under the mean setting. Explorative early, converges to the mean.
At the end of an episode, most correctors' offsets are pinned down; the downstream-most ones (seen
by few monitors, or compensating each other; wide vertical beam) stay uncertain, but the
environment is still solved.

## 4 Evaluation
Same environments and initial settings for all agents, background ν = 0.25, horizon 150 steps,
solved at mean normalized symmetry > 0.95. 20 randomized environments × 5 seeds = 100 episodes
per agent. Bayes: 10⁵ particles, roughening 0.5, α = 0.1, β = 0.6, always built on the nominal
model. PPO: one seed unrandomized (4·10⁷ steps), one with domain randomization (10⁸ steps).

**Table 2 — solved episodes (mean steps of solved episodes)**

| Agent | off | low | medium | high | high + Student-t |
|---|---|---|---|---|---|
| Bayes | 99/100 (32) | 99/100 (29) | 99/100 (26) | 98/100 (32) | 91/100 (45) |
| PPO | 100/100 (13) | 100/100 (17) | 83/100 (24) | 58/100 (25) | 46/100 (30) |
| PPO, randomized | 100/100 (12) | 100/100 (12) | 100/100 (12) | 95/100 (14) | 94/100 (23) |

- Unrandomized: everyone solves nearly all; PPO ~12–13 steps vs Bayes ~32.
- Randomized: nominal-model PPO degrades; Bayes stays 91–99/100 because the soft likelihood
  tolerates model discrepancy. Randomized PPO is the most robust and 2–3× faster, but needs 10⁸
  simulated steps of training; Bayes needs none but ~30 s CPU per episode (PPO < 1 s).

## 5 Conclusion
Both approaches steer the simulated line and handle randomized optics. RL is slightly more
successful and faster; Bayes is interpretable (belief can diagnose the model) and could improve
with tuning. Sim-to-real gaps remain (Gaussian beam, centre-only collision detection, uniform
noise).

## Key references
[1] A. Menor de Oñate et al., "Trajectory steering for DC beams at the CERN SPS using RL based on
intensity measurements", IPAC'25, THPM113, doi:10.18429/JACoW-IPAC2025-THPM113.
[8] Gordon, Salmond, Smith, "Novel approach to nonlinear/non-Gaussian Bayesian state estimation",
IEE Proc. F 140(2), 1993 (particle filter). [9] Huber 1964 (Huber loss). [10] Thompson 1933.
