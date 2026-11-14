import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional, List
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


@dataclass
class BioreactorDifficultyConfig:
    level: int
    max_steps: int
    dt_hours: float
    initial_glucose: float         # g/L
    max_glucose_feed: float        # g/L/hr
    acetaldehyde_tox: float        # g/L lethal threshold
    acetate_tox: float             # g/L base lethal threshold at pH 7.0
    burden_sensitivity: float      # metabolic burden penalty
    cooling_capacity: float        # max jacket cooling power (°C/hr)
    kLa_max: float                 # max oxygen mass transfer coeff (1/hr)
    disturbance_window: Tuple[int, int]  # scheduled cooling efficiency drop (start_hr, end_hr)
    utility_cost_mult: float       # multiplier for cooling, aeration, and base costs


DIFFICULTY_TIERS: Dict[int, BioreactorDifficultyConfig] = {
    1: BioreactorDifficultyConfig(1, 24, 1.0, 20.0, 15.0, 1.20, 8.0, 0.08, 12.0, 45.0, (99, 99), 0.5),
    2: BioreactorDifficultyConfig(2, 30, 1.0, 15.0, 18.0, 0.85, 6.0, 0.12, 10.0, 38.0, (14, 17), 0.8),
    3: BioreactorDifficultyConfig(3, 36, 1.0, 10.0, 20.0, 0.60, 4.5, 0.16, 8.5,  32.0, (16, 21), 1.0),
    4: BioreactorDifficultyConfig(4, 42, 1.0, 8.0,  22.0, 0.42, 3.2, 0.22, 7.2,  26.0, (18, 25), 1.3),
    5: BioreactorDifficultyConfig(5, 48, 1.0, 5.0,  25.0, 0.30, 2.2, 0.28, 6.0,  21.0, (20, 30), 1.6),
}


class CoupledEColiBioreactorEnv:
    """
    Fully observable deterministic fed-batch E. coli bioethanol bioreactor with coupled
    biochemical kinetics, Dissolved Oxygen (DO) mass transfer, pH dynamics, and thermal balance.
    """

    def __init__(self, difficulty: int = 3):
        if difficulty not in DIFFICULTY_TIERS:
            raise ValueError(f"Difficulty must be in {list(DIFFICULTY_TIERS.keys())}")
        self.cfg = DIFFICULTY_TIERS[difficulty]
        self.reset()

    def reset(self) -> Dict[str, Any]:
        self.step_count = 0
        self.cumulative_glucose = self.cfg.initial_glucose
        self.cumulative_inducer = 0.0
        self.cumulative_utility_cost = 0.0
        self.terminated = False
        self.crash_reason: Optional[str] = None

        # State vector: [X, G, DO, Pyr, AcAld, Acet, EtOH, Temp, pH, Viability]
        self.state = np.array([
            0.5,                       # X: Biomass (g/L)
            self.cfg.initial_glucose,  # G: Glucose (g/L)
            100.0,                     # DO: Dissolved Oxygen (% saturation)
            0.05,                      # Pyr: Pyruvate pool (g/L)
            0.0,                       # AcAld: Toxic Acetaldehyde (g/L)
            0.0,                       # Acet: Toxic Acetate (g/L)
            0.0,                       # EtOH: Bioethanol product (g/L)
            37.0,                      # Temp: Reactor temperature (°C)
            7.0,                       # pH: Broth pH
            1.0,                       # Viability: [0.0 - 1.0]
        ], dtype=np.float64)

        return self.get_observation()

    def _get_cooling_efficiency(self, current_step: int) -> float:
        start_h, end_h = self.cfg.disturbance_window
        return 0.45 if start_h <= current_step < end_h else 1.0

    def _odes(self, t: float, y: np.ndarray, action: Tuple[float, ...], cool_eff: float) -> np.ndarray:
        X, G, DO, Pyr, AcAld, Acet, EtOH, Temp, pH, V = y
        X, G, DO, Pyr, AcAld, Acet, EtOH, V = np.maximum([X, G, DO, Pyr, AcAld, Acet, EtOH, V], 0.0)
        g_feed, o2_sparge, pdc_expr, adh_expr, cooling_power, base_add = action

        # --- A. Temperature & pH Kinetic Modifiers ---
        # Growth rate peaks at 37°C; drops sharply above 40°C (heat shock)
        temp_growth_factor = np.exp(-0.045 * (Temp - 37.0) ** 2) if Temp <= 39.5 else np.exp(-0.35 * (Temp - 37.0) ** 2)
        # Synthetic enzyme folding/stability prefers slightly cooler temps (28-33°C)
        enzyme_stability = np.exp(-0.025 * (Temp - 31.0) ** 2)
        # Enzyme & cellular activity peaks at pH 7.0
        ph_factor = np.exp(-1.8 * (pH - 7.0) ** 2)

        # --- B. Substrate Uptake & Glycolysis ---
        etoh_inhib = np.exp(-0.025 * EtOH)
        q_g = 1.45 * (G / (G + 0.5)) * temp_growth_factor * ph_factor * etoh_inhib * V
        v_glyc = 0.95 * q_g * X

        # --- C. Oxygen Mass Transfer (OTR vs OUR) ---
        # Higher temperature lowers O2 solubility; high biomass increases viscosity & O2 demand
        do_sat = max(40.0, 100.0 - 1.2 * (Temp - 25.0))
        viscosity_penalty = 1.0 / (1.0 + 0.015 * X)
        otr = self.cfg.kLa_max * o2_sparge * viscosity_penalty * (do_sat - DO)

        o2_affinity = DO / (DO + 8.0)
        v_resp_cap = 0.65 * o2_affinity * temp_growth_factor * ph_factor * X * V

        # --- D. Synthetic Ethanol Pathway (PET operon) ---
        v_pdc = 0.90 * pdc_expr * enzyme_stability * ph_factor * (Pyr / (Pyr + 0.15)) * X * V
        v_adh = 1.20 * adh_expr * enzyme_stability * ph_factor * (AcAld / (AcAld + 0.08)) * X * V

        pyr_avail = v_glyc + 2.0 * Pyr
        v_pdc_actual = min(v_pdc, 0.85 * pyr_avail)
        rem_pyr = max(0.0, v_glyc - v_pdc_actual)

        v_resp = min(rem_pyr, v_resp_cap)
        v_acet = max(0.0, rem_pyr - v_resp) * 0.65  # Overflow metabolism into acetic acid

        # --- E. Coupled Toxicity & Metabolic Burden ---
        # Weak-acid effect: Acetate becomes exponentially more toxic at low pH (protonated form)
        acid_amplification = 1.0 + max(0.0, 6.8 - pH) * 2.5
        effective_acet_tox = self.cfg.acetate_tox / acid_amplification

        burden = self.cfg.burden_sensitivity * (pdc_expr**1.5 + adh_expr**1.5) * (1.0 + 0.05 * max(0.0, Temp - 32.0))
        thermal_stress = max(0.0, (Temp - 39.5) / 3.5) ** 2 + max(0.0, (22.0 - Temp) / 5.0) ** 2
        ph_stress = max(0.0, abs(pH - 7.0) - 0.45) ** 2 * 3.0

        tox_stress = (
            max(0.0, AcAld / self.cfg.acetaldehyde_tox) ** 2
            + max(0.0, Acet / effective_acet_tox) ** 2
            + max(0.0, EtOH / 65.0) ** 2
            + thermal_stress
            + ph_stress
        )

        # --- F. Differential Equations ---
        mu = max(0.0, 0.50 * (v_resp / (X + 1e-6)) - burden - 0.15 * tox_stress)
        dX = (mu - 0.025 * tox_stress) * X
        dG = g_feed - q_g * X
        dDO = otr - 22.0 * v_resp
        dPyr = v_glyc - v_pdc_actual - v_resp - (v_acet / 0.65)
        dAcAld = 0.50 * v_pdc_actual - v_adh
        dAcet = v_acet - 0.05 * o2_affinity * (Acet / (Acet + 0.5)) * X
        dEtOH = 1.04 * v_adh

        # Thermal balance: Metabolic heat generation - Jacket cooling + Ambient drift (25°C)
        heat_gen = 0.55 * v_resp + 0.25 * v_glyc
        heat_removed = self.cfg.cooling_capacity * cooling_power * cool_eff * ((Temp - 15.0) / 22.0)
        ambient_exchange = 0.08 * (25.0 - Temp)
        dTemp = heat_gen - heat_removed + ambient_exchange

        # pH balance: Acidification from acetate & CO2 respiration vs. Base dosing & buffer capacity
        buffer_capacity = 1.0 + 0.04 * X
        acid_load = (0.38 * v_acet + 0.04 * v_resp) / buffer_capacity
        base_effect = (1.8 * base_add) / buffer_capacity
        dpH = base_effect - acid_load

        # Viability decay
        dV = -0.20 * max(0.0, tox_stress - 0.35) - 0.04 * max(0.0, burden - 0.35)
        if V >= 1.0 and dV > 0:
            dV = 0.0

        return np.array([dX, dG, dDO, dPyr, dAcAld, dAcet, dEtOH, dTemp, dpH, dV])

    def step(self, action_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Action dictionary:
          - glucose_feed: float in [0.0, max_glucose_feed] (g/L/hr)
          - o2_sparge: float in [0.0, 1.0] (agitation & aeration intensity)
          - pdc_expression: float in [0.0, 2.0] (Pyruvate Decarboxylase induction)
          - adh_expression: float in [0.0, 2.0] (Alcohol Dehydrogenase induction)
          - cooling_power: float in [0.0, 1.0] (cooling jacket valve)
          - base_addition: float in [0.0, 1.0] (NH4OH/NaOH pH buffer dosing)
          - harvest: bool (optional early harvest)
        """
        if self.terminated:
            raise RuntimeError("Episode already terminated. Call reset().")

        g_feed = float(np.clip(action_dict.get("glucose_feed", 0.0), 0.0, self.cfg.max_glucose_feed))
        o2_sparge = float(np.clip(action_dict.get("o2_sparge", 0.5), 0.0, 1.0))
        pdc_expr = float(np.clip(action_dict.get("pdc_expression", 0.0), 0.0, 2.0))
        adh_expr = float(np.clip(action_dict.get("adh_expression", 0.0), 0.0, 2.0))
        cooling = float(np.clip(action_dict.get("cooling_power", 0.3), 0.0, 1.0))
        base_add = float(np.clip(action_dict.get("base_addition", 0.0), 0.0, 1.0))
        harvest_now = bool(action_dict.get("harvest", False))

        cool_eff = self._get_cooling_efficiency(self.step_count)
        act_tuple = (g_feed, o2_sparge, pdc_expr, adh_expr, cooling, base_add)

        sol = solve_ivp(
            self._odes,
            [0.0, self.cfg.dt_hours],
            self.state,
            args=(act_tuple, cool_eff),
            method="RK45",
            rtol=1e-5,
            atol=1e-7,
        )
        self.state = sol.y[:, -1]
        self.state[:7] = np.maximum(self.state[:7], 0.0)
        self.state[2] = np.clip(self.state[2], 0.0, 100.0)
        self.state[7] = np.clip(self.state[7], 15.0, 55.0)
        self.state[8] = np.clip(self.state[8], 3.5, 10.5)
        self.state[9] = np.clip(self.state[9], 0.0, 1.0)

        self.step_count += 1
        self.cumulative_glucose += g_feed * self.cfg.dt_hours
        self.cumulative_inducer += (pdc_expr + adh_expr) * self.cfg.dt_hours
        step_utility = self.cfg.utility_cost_mult * (0.25 * o2_sparge + 0.30 * cooling + 0.35 * base_add)
        self.cumulative_utility_cost += step_utility

        X, G, DO, Pyr, AcAld, Acet, EtOH, Temp, pH, V = self.state

        if V < 0.20:
            self.terminated = True
            self.crash_reason = self._diagnose_crash(AcAld, Acet, Temp, pH, pdc_expr, adh_expr)
        elif X < 0.05:
            self.terminated = True
            self.crash_reason = "Culture washout / biomass collapse"
        elif harvest_now or self.step_count >= self.cfg.max_steps:
            self.terminated = True

        reward = 0.0
        if self.terminated:
            if self.crash_reason is not None:
                reward = -15.0
            else:
                gross_revenue = 1.65 * EtOH
                costs = (0.12 * self.cumulative_glucose) + (0.10 * self.cumulative_inducer) + self.cumulative_utility_cost
                reward = float(gross_revenue - costs)

        info = {"crash_reason": self.crash_reason, "step": self.step_count}
        return self.get_observation(), reward, self.terminated, info

    def _diagnose_crash(self, AcAld: float, Acet: float, Temp: float, pH: float, pdc: float, adh: float) -> str:
        if Temp > 41.0:
            return f"Thermal runaway death ({Temp:.1f}°C > 41.0°C): metabolic heat exceeded jacket cooling."
        if pH < 5.6:
            return f"Acidification crash (pH {pH:.2f} < 5.6): unbuffered acetate/CO2 protonated weak acids."
        if pH > 8.2:
            return f"Alkaline shock crash (pH {pH:.2f} > 8.2): excessive base addition."
        if AcAld > self.cfg.acetaldehyde_tox * 0.8:
            return f"Acetaldehyde toxicity ({AcAld:.2f} g/L > {self.cfg.acetaldehyde_tox} g/L): pdc exceeded adhB."
        if Acet > self.cfg.acetate_tox * 0.65:
            return f"Acetate overflow toxicity ({Acet:.2f} g/L at pH {pH:.2f}): O2 transfer bottleneck or excess glucose."
        return f"Metabolic overexpression burden collapse (pdc={pdc:.2f}, adhB={adh:.2f}, Temp={Temp:.1f}°C)."

    def get_observation(self) -> Dict[str, Any]:
        X, G, DO, Pyr, AcAld, Acet, EtOH, Temp, pH, V = self.state
        start_h, end_h = self.cfg.disturbance_window
        return {
            "step": self.step_count,
            "steps_remaining": self.cfg.max_steps - self.step_count,
            "state": {
                "biomass_g_L": round(float(X), 3),
                "glucose_g_L": round(float(G), 3),
                "dissolved_o2_pct": round(float(DO), 2),
                "temperature_C": round(float(Temp), 2),
                "pH": round(float(pH), 3),
                "pyruvate_g_L": round(float(Pyr), 3),
                "acetaldehyde_g_L": round(float(AcAld), 3),
                "acetate_g_L": round(float(Acet), 3),
                "bioethanol_g_L": round(float(EtOH), 3),
                "viability": round(float(V), 3),
            },
            "reactor_conditions": {
                "current_cooling_efficiency": self._get_cooling_efficiency(self.step_count),
                "scheduled_cooling_maintenance_hours": [start_h, end_h] if start_h < self.cfg.max_steps else None,
                "kLa_max_hr": self.cfg.kLa_max,
            },
            "operating_limits": {
                "optimal_temp_growth_C": 37.0,
                "optimal_temp_enzyme_C": 31.0,
                "lethal_temp_high_C": 41.5,
                "optimal_pH_range": [6.8, 7.2],
                "lethal_pH_range": [5.6, 8.2],
                "acetaldehyde_lethal_g_L": self.cfg.acetaldehyde_tox,
                "acetate_base_lethal_g_L": self.cfg.acetate_tox,
                "min_viability": 0.20,
            },
            "action_bounds": {
                "glucose_feed": [0.0, self.cfg.max_glucose_feed],
                "o2_sparge": [0.0, 1.0],
                "pdc_expression": [0.0, 2.0],
                "adh_expression": [0.0, 2.0],
                "cooling_power": [0.0, 1.0],
                "base_addition": [0.0, 1.0],
            },
            "terminated": self.terminated,
            "crash_reason": self.crash_reason,
        }


# =====================================================================
# Reproducible Closed-Loop Reference Solver (MPC / Biphasic Controller)
# =====================================================================
def solve_reference_policy(difficulty: int = 3) -> Dict[str, Any]:
    """
    Optimizes a biphasic strategy combined with closed-loop feedback rules for
    pH buffering, thermal regulation (including pre-cooling before maintenance),
    and DO-stat glucose throttling.
    """
    env = CoupledEColiBioreactorEnv(difficulty=difficulty)
    max_steps = env.cfg.max_steps
    max_g = env.cfg.max_glucose_feed

    def controller_action(obs: Dict[str, Any], p: np.ndarray, step: int) -> Dict[str, Any]:
        switch_frac, g1, pdc1, adh1, g2, pdc2, adh2, kp_cool, kp_base = p
        st = obs["state"]
        switch_step = int(np.clip(switch_frac, 0.15, 0.85) * max_steps)

        # 1. Phase setpoints (Growth @ 37°C -> Production @ 31°C)
        if step < switch_step:
            target_temp = 37.0
            base_g, pdc, adh = g1, pdc1, adh1
        else:
            target_temp = 31.0
            base_g, pdc, adh = g2, pdc2, adh2

        # Pre-cool 2 hours ahead of scheduled cooling maintenance
        maint = obs["reactor_conditions"]["scheduled_cooling_maintenance_hours"]
        if maint and (maint[0] - 2 <= step < maint[1]):
            target_temp -= 2.5

        # 2. Closed-loop pH, Temperature & DO-Stat safety overrides
        pH_err = 7.05 - st["pH"]
        base_add = float(np.clip(kp_base * max(0.0, pH_err) + 0.08 * st["acetate_g_L"], 0.0, 1.0))

        temp_err = st["temperature_C"] - target_temp
        cooling = float(np.clip(0.25 + kp_cool * temp_err, 0.0, 1.0))

        # DO-stat throttling: cut glucose feed if DO drops below 22% to avoid acetate cascade
        do_scale = np.clip((st["dissolved_o2_pct"] - 10.0) / 20.0, 0.15, 1.0)
        g_feed = float(np.clip(base_g * do_scale, 0.0, max_g))
        o2_sparge = float(np.clip(0.35 + (60.0 - st["dissolved_o2_pct"]) / 50.0, 0.2, 1.0))

        return {
            "glucose_feed": round(g_feed, 3),
            "o2_sparge": round(o2_sparge, 3),
            "pdc_expression": round(float(pdc), 3),
            "adh_expression": round(float(adh), 3),
            "cooling_power": round(cooling, 3),
            "base_addition": round(base_add, 3),
            "harvest": (step == max_steps - 1),
        }

    def objective(p: np.ndarray) -> float:
        obs = env.reset()
        total_r = 0.0
        for s in range(max_steps):
            act = controller_action(obs, p, s)
            obs, r, done, _ = env.step(act)
            total_r += r
            if done:
                break
        return -total_r

    x0 = np.array([0.45, max_g * 0.45, 0.15, 0.25, max_g * 0.70, 1.25, 1.50, 0.25, 0.80])
    bounds = [
        (0.15, 0.85), (0.0, max_g), (0.0, 2.0), (0.0, 2.0),
        (0.0, max_g), (0.0, 2.0), (0.0, 2.0), (0.05, 0.8), (0.1, 2.0),
    ]
    res = minimize(objective, x0, method="Powell", bounds=bounds, options={"maxiter": 250})

    # Replay optimal trajectory
    obs = env.reset()
    trajectory: List[Dict[str, Any]] = []
    final_reward = 0.0
    for s in range(max_steps):
        act = controller_action(obs, res.x, s)
        next_obs, r, done, _ = env.step(act)
        trajectory.append({"obs": obs, "action": act, "reward": r})
        obs = next_obs
        final_reward += r
        if done:
            break

    return {
        "difficulty": difficulty,
        "reference_reward": round(final_reward, 3),
        "final_state": obs["state"],
        "crash_reason": obs["crash_reason"],
        "trajectory": trajectory,
    }


if __name__ == "__main__":
    for lvl in [1, 3, 5]:
        out = solve_reference_policy(difficulty=lvl)
        st = out["final_state"]
        print(
            f"[Lvl {lvl}] Reward: {out['reference_reward']:>6.2f} | "
            f"EtOH: {st['bioethanol_g_L']:>5.2f} g/L | X: {st['biomass_g_L']:>5.2f} g/L | "
            f"Temp: {st['temperature_C']}°C | pH: {st['pH']} | DO: {st['dissolved_o2_pct']}% | "
            f"Viab: {st['viability']}"
        )
