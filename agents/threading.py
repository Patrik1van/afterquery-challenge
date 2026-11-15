"""Scripted threading reference agent for TT20 transfer line (operator method).

Features:
- Progressive monitor-by-monitor beam threading along the line.
- Adjusts both horizontal and vertical planes in the same shot.
- Scans corrector angle over a few shots until monitor level indicates beam on foils.
- Centres beam on foils using inverse foil model x = -sigma * Phi^{-1}(r) and nominal response matrix.
- Keeps upstream correctors fixed during threading.
- Uses all 60 shots: finishes with damped regularized least-squares refinement on all visible monitors.
- Final shot commands the best setting found by measured symmetry from all observations.
"""

from __future__ import annotations

from typing import Any
import numpy as np
from scipy.special import ndtri

from beamsim.env_api import BeamlineEnv


def compute_measured_symmetry(obs_h: list[dict[str, Any]], obs_v: list[dict[str, Any]]) -> float:
    """Compute intensity-normalized symmetry from noisy monitor readings."""
    syms = []
    for o in obs_h:
        r = o["ratio"]
        l = o["level"]
        s_raw = np.sqrt(max(0.0, 1.0 - abs(2.0 * r - 1.0)))
        s_hat = s_raw * min(1.0, l)
        syms.append(s_hat)
    for o in obs_v:
        r = o["ratio"]
        l = o["level"]
        s_raw = np.sqrt(max(0.0, 1.0 - abs(2.0 * r - 1.0)))
        s_hat = s_raw * min(1.0, l)
        syms.append(s_hat)
    return float(np.mean(syms))


def invert_foil_x(ratio: float, sigma: float) -> float:
    """Invert foil model for position x near centre: r = Phi(-x/sigma)."""
    r_clipped = float(np.clip(ratio, 0.01, 0.99))
    return float(-sigma * ndtri(r_clipped))


class ThreadingAgent:
    """Scripted threading reference agent using classical operator beamline threading."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def solve(
        self,
        env: BeamlineEnv,
        on_shot_callback: Any = None,
    ) -> list[dict[str, Any]]:
        """Run the threading agent loop on the environment until submit or budget exhausted."""
        info = env.get_machine_info()
        theta_max = float(info["theta_max"])
        budget = int(info["budget"])

        r_sem_h = np.array(info["r_sem_h"], dtype=np.float64)  # (11, 10)
        r_sem_v = np.array(info["r_sem_v"], dtype=np.float64)  # (10, 12)
        sigma_h = np.array(info["h"]["sigma_sem"], dtype=np.float64)
        sigma_v = np.array(info["v"]["sigma_sem"], dtype=np.float64)

        theta_h = np.zeros(10, dtype=np.float64)
        theta_v = np.zeros(12, dtype=np.float64)

        best_score = -1.0
        best_theta_h = theta_h.copy()
        best_theta_v = theta_v.copy()
        shots_used = 0
        agent_logs: list[dict[str, Any]] = []

        def step(th_h: np.ndarray, th_v: np.ndarray, action_type: str = "step") -> dict[str, Any]:
            nonlocal shots_used, best_score, best_theta_h, best_theta_v
            act_h = np.clip(th_h, -theta_max, theta_max)
            act_v = np.clip(th_v, -theta_max, theta_max)
            obs = env.set_correctors(act_h.tolist(), act_v.tolist())
            shots_used += 1

            s_meas = compute_measured_symmetry(obs["h"], obs["v"])
            if s_meas > best_score:
                best_score = s_meas
                best_theta_h = act_h.copy()
                best_theta_v = act_v.copy()

            log_entry = {
                "shot": shots_used,
                "theta_h": act_h.tolist(),
                "theta_v": act_v.tolist(),
                "obs_h": obs["h"],
                "obs_v": obs["v"],
                "measured_symmetry": s_meas,
                "action_type": action_type,
            }
            agent_logs.append(log_entry)
            if on_shot_callback is not None:
                on_shot_callback(shots_used, act_h, act_v, s_meas)
            return obs

        # Shot 1: Initial state at theta = 0
        obs = step(theta_h, theta_v, action_type="initial_zero")

        # Stage definition: (sem_idx_h, corr_idx_h, sem_idx_v, corr_idx_v)
        stages: list[tuple[int | None, int | None, int | None, int | None]] = [
            (0, 0, 0, 0),
            (1, 1, 1, 1),
            (2, 2, 2, 2),
            (3, 3, 3, 3),
            (4, 4, 4, 4),
            (5, 5, 5, 5),
            (6, 6, 6, 6),
            (7, 7, 7, 7),
            (10, 8, 8, 8),
            (10, 9, 8, 9),
            (None, None, 9, 10),
            (None, None, 9, 11),
        ]

        scan_grid = [
            -0.35e-3, 0.35e-3,
            -0.20e-3, 0.20e-3,
            -0.44e-3, 0.44e-3,
            -0.10e-3, 0.10e-3,
        ]

        # Phase 1: Monitor-by-monitor threading
        for m_h, c_h, m_v, c_v in stages:
            if shots_used >= 45:
                break

            h_needed = (m_h is not None and c_h is not None)
            v_needed = (m_v is not None and c_v is not None)

            need_scan_h = h_needed and (obs["h"][m_h]["level"] < 0.50 or abs(obs["h"][m_h]["ratio"] - 0.5) > 0.25)
            need_scan_v = v_needed and (obs["v"][m_v]["level"] < 0.50 or abs(obs["v"][m_v]["ratio"] - 0.5) > 0.25)

            if need_scan_h or need_scan_v:
                best_lvl_h = obs["h"][m_h]["level"] if h_needed else 1.0
                best_th_h_stage = theta_h[c_h] if h_needed else 0.0
                best_lvl_v = obs["v"][m_v]["level"] if v_needed else 1.0
                best_th_v_stage = theta_v[c_v] if v_needed else 0.0

                for scan_val in scan_grid:
                    if shots_used >= 45:
                        break
                    trial_h = theta_h.copy()
                    trial_v = theta_v.copy()

                    if need_scan_h:
                        trial_h[c_h] = np.clip(scan_val, -theta_max, theta_max)
                    if need_scan_v:
                        trial_v[c_v] = np.clip(scan_val, -theta_max, theta_max)

                    obs = step(trial_h, trial_v, action_type=f"scan_stage_{m_h}_{m_v}")

                    if need_scan_h:
                        lvl_h = obs["h"][m_h]["level"]
                        if lvl_h > best_lvl_h:
                            best_lvl_h = lvl_h
                            best_th_h_stage = trial_h[c_h]
                        if lvl_h >= 0.55:
                            need_scan_h = False

                    if need_scan_v:
                        lvl_v = obs["v"][m_v]["level"]
                        if lvl_v > best_lvl_v:
                            best_lvl_v = lvl_v
                            best_th_v_stage = trial_v[c_v]
                        if lvl_v >= 0.55:
                            need_scan_v = False

                    if not need_scan_h and not need_scan_v:
                        break

                if h_needed:
                    theta_h[c_h] = best_th_h_stage
                if v_needed:
                    theta_v[c_v] = best_th_v_stage
                obs = step(theta_h, theta_v, action_type=f"scan_best_{m_h}_{m_v}")

            # Centering step using foil ratio inversion (up to 2 iterations)
            for _ in range(2):
                if shots_used >= 50:
                    break
                changed = False
                if h_needed and obs["h"][m_h]["level"] >= 0.35 and abs(r_sem_h[m_h, c_h]) > 1e-3 and abs(obs["h"][m_h]["ratio"] - 0.5) > 0.05:
                    x_est_h = invert_foil_x(obs["h"][m_h]["ratio"], sigma_h[m_h])
                    theta_h[c_h] = np.clip(theta_h[c_h] + x_est_h / r_sem_h[m_h, c_h], -theta_max, theta_max)
                    changed = True
                if v_needed and obs["v"][m_v]["level"] >= 0.35 and abs(r_sem_v[m_v, c_v]) > 1e-3 and abs(obs["v"][m_v]["ratio"] - 0.5) > 0.05:
                    x_est_v = invert_foil_x(obs["v"][m_v]["ratio"], sigma_v[m_v])
                    theta_v[c_v] = np.clip(theta_v[c_v] + x_est_v / r_sem_v[m_v, c_v], -theta_max, theta_max)
                    changed = True
                if changed:
                    obs = step(theta_h, theta_v, action_type=f"center_stage_{m_h}_{m_v}")
                else:
                    break
        # Phase 2: Least-squares refinement on all visible monitors
        damping = 0.5
        reg_lambda = 1e-3

        while shots_used < budget - 1:
            vis_h = [i for i, o in enumerate(obs["h"]) if o["level"] >= 0.35]
            vis_v = [i for i, o in enumerate(obs["v"]) if o["level"] >= 0.35]

            dth_h = np.zeros(10, dtype=np.float64)
            dth_v = np.zeros(12, dtype=np.float64)

            if len(vis_h) > 0:
                x_vis_h = np.array([invert_foil_x(obs["h"][i]["ratio"], sigma_h[i]) for i in vis_h])
                R_sub_h = r_sem_h[vis_h, :]
                lhs_h = R_sub_h.T @ R_sub_h + reg_lambda * np.eye(10)
                rhs_h = R_sub_h.T @ x_vis_h
                try:
                    dth_h = np.linalg.solve(lhs_h, rhs_h)
                except np.linalg.LinAlgError:
                    dth_h = np.linalg.pinv(lhs_h) @ rhs_h

            if len(vis_v) > 0:
                x_vis_v = np.array([invert_foil_x(obs["v"][i]["ratio"], sigma_v[i]) for i in vis_v])
                R_sub_v = r_sem_v[vis_v, :]
                lhs_v = R_sub_v.T @ R_sub_v + reg_lambda * np.eye(12)
                rhs_v = R_sub_v.T @ x_vis_v
                try:
                    dth_v = np.linalg.solve(lhs_v, rhs_v)
                except np.linalg.LinAlgError:
                    dth_v = np.linalg.pinv(lhs_v) @ rhs_v

            cand_th_h = np.clip(theta_h + damping * dth_h, -theta_max, theta_max)
            cand_th_v = np.clip(theta_v + damping * dth_v, -theta_max, theta_max)

            obs = step(cand_th_h, cand_th_v, action_type="least_squares_refinement")
            theta_h = cand_th_h
            theta_v = cand_th_v

        # Phase 3: Final shot (Shot 60) = best setting found by measured symmetry
        if shots_used < budget:
            obs = step(best_theta_h, best_theta_v, action_type="final_best_measured")

        env.submit()
        return agent_logs


