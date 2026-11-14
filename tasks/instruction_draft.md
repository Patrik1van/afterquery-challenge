# Operator Briefing: CERN TT20 Beam Steering Task

You are an expert accelerator operator controlling a simulated proton transfer line (~900 meters long) modeled after the CERN TT20 beamline.

## 1. Goal
Your mission is to steer, thread, and centre the proton beam across the entire transfer line through all Secondary Emission Monitors (SEMs) in both the horizontal and vertical planes within a limited budget of **60 shots**.

An episode is **solved** when the true mean intensity-normalized symmetry $S_{\text{mean}} \ge 0.95$.

### Reward Formula
The evaluation score is computed server-side from true, noise-free state:
- $S_0$: noise-free mean symmetry at initial state $\theta = 0$ (typically near 0 due to missteered beam).
- $S_f$: noise-free mean symmetry at your final submitted setting.
- $\text{progress} = \operatorname{clip}\left(\frac{S_f - S_0}{0.95 - S_0}, 0.0, 1.0\right)$
- $\text{solved} = (S_f \ge 0.95)$
- $\text{reward} = 0.8 \cdot \text{progress} + 0.2 \cdot \text{solved} \cdot \left(1 - \frac{\text{shots\_used}}{\text{budget}}\right)$

Notice that doing nothing gives 0 reward. Making real progress towards threading and centring gives up to 0.8, and solving with fewer shots earns an efficiency bonus up to 0.2.

---

## 2. Physics in Brief
1. **Response Matrix & Trajectory**:
   Beam deflection is governed by linear optics per plane:
   $$x_h = R_h (\Delta\theta_h - \theta_h), \qquad x_v = R_v (\Delta\theta_v - \theta_v)$$
   where $\Delta\theta$ are unknown hidden missteering kicks, and $\theta$ are your commanded corrector dipole angles.
   - Horizontal: 10 correctors, 11 SEM monitors.
   - Vertical: 12 correctors, 10 SEM monitors.
   - The last 1–2 correctors in each plane only influence the final monitor.

2. **Secondary Emission Monitors (SEMs)**:
   Each SEM monitor has two split foils spanning $[-L, 0]$ and $[0, L]$ with $L \approx 10\sigma$.
   - **Ratio** $r = \tilde{I}_1 / (\tilde{I}_1 + \tilde{I}_2)$: Measures relative beam position across the two foils. For a centred beam ($x = 0$), $r \approx 0.5$. If beam is deflected positive ($x > 0$), $r < 0.5$; if $x < 0$, $r > 0.5$.
   - **Level** $\ell = (\tilde{I}_1 + \tilde{I}_2) / I_{\max}$: Measures total intercepted intensity relative to a centred beam. When beam passes through the monitor, $\ell \approx 1.0$. When beam misses foils or is lost upstream, the foils read only background noise ($\ell \approx \nu \approx 0.25$).

3. **Joint Beam Loss**:
   The beampipe has aperture half-widths $a_h, a_v \sim 25\text{–}35\text{ mm}$. If $|x_h| > a_h$ OR $|x_v| > a_v$ at any longitudinal point along the line, the beam hits the wall and is lost. **All monitors in both planes downstream of the loss point detect background noise only.**

4. **Nominal Model vs Real Machine**:
   The response matrices and beam sizes provided by `get_machine_info()` are **nominal**. The true machine operates with realistic domain mismatch (slight corrector gain errors ~±2%, optical beta-beating ~30–50%, background noise variations). Do not rely solely on open-loop inversion—use observed monitor readings to iteratively refine your steering.

---

## 3. Available Tools
You have access to 5 tools:

1. **`get_machine_info()`** (Costs 0 shots):
   Returns the nominal model: element coordinates, nominal Twiss parameters, nominal response matrices $R_{\text{sem}}$ and $R_{\text{ap}}$ for both planes, apertures, noise level $\nu=0.25$, setting bound $\theta_{\max}$, budget (60), and remaining shots.
2. **`set_correctors(h: list[float], v: list[float])`** (Costs 1 shot):
   Applies absolute corrector dipole deflection angles (10 floats for horizontal, 12 floats for vertical) clipped to $[-\theta_{\max}, \theta_{\max}]$. Returns noisy monitor readings $\{r, \ell\}$ for each SEM in H and V, plus `shots_remaining`.
3. **`get_history()`** (Costs 0 shots):
   Returns the complete chronological log of all your previous settings and the corresponding monitor readings.
4. **`submit()`** (Costs 0 shots):
   Freezes the episode when you are satisfied that the beam is centred. No further corrector commands are accepted. Your last commanded setting is evaluated for the final score.
5. **`run_python(code: str)`** (Costs 0 shots):
   Executes Python code in an isolated subprocess with NumPy and SciPy available (30 s timeout). You can write scripts to invert response matrices (e.g. SVD / pseudo-inverse), fit trajectories, run particle filters, or optimize settings.

---

## 4. Operational Strategy
- Start by calling `get_machine_info()` and examining the initial state.
- Write Python scripts via `run_python` to compute your candidate corrections based on the response matrices and monitor readings.
- Progressively thread the beam from upstream monitors downstream: correct early monitors first so the beam reaches downstream monitors.
- Once the beam reaches all monitors, fine-tune the correctors to bring ratios $r \to 0.5$ and levels $\ell \to 1.0$.
- When confident, call `submit()`.
