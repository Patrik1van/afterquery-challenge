"""Agent-facing environment API matching future MCP tools.

Strictly encapsulates the hidden state (true offsets Delta_theta, corrector gains g,
perturbed optics, true beam positions and score tokens). Only nominal machine
parameters are reachable through this interface.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from beamsim.lattice import Lattice, load_lattice
from beamsim.randomize import compute_response_matrix
from beamsim.scoring import ScoreDetails, compute_reward
from beamsim.sim import BeamlineSim


class BeamlineEnv:
    """Agent environment wrapper matching future MCP tools."""

    def __init__(
        self,
        seed: int = 42,
        budget: int = 40,
        randomization: str = "medium",
        lattice: Lattice | None = None,
        theta_max: float | None = None,
    ) -> None:
        self._lattice = lattice if lattice is not None else load_lattice()
        self._theta_max = theta_max if theta_max is not None else self._lattice.theta_max
        self._budget = int(budget)
        self._seed = int(seed)
        self._randomization = str(randomization)

        self._sim = BeamlineSim(
            lattice=self._lattice,
            seed=self._seed,
            spread=self._randomization,
            theta_max=self._theta_max,
        )

        self._submitted = False
        self._history: list[dict[str, Any]] = []

        self._nominal_r_sem_h = compute_response_matrix(
            self._lattice.h.beta_sem, self._lattice.h.mu_sem,
            self._lattice.h.beta_corr, self._lattice.h.mu_corr,
        )
        self._nominal_r_ap_h = compute_response_matrix(
            self._lattice.beta_h_ap, self._lattice.mu_h_ap,
            self._lattice.h.beta_corr, self._lattice.h.mu_corr,
        )
        self._nominal_r_sem_v = compute_response_matrix(
            self._lattice.v.beta_sem, self._lattice.v.mu_sem,
            self._lattice.v.beta_corr, self._lattice.v.mu_corr,
        )
        self._nominal_r_ap_v = compute_response_matrix(
            self._lattice.beta_v_ap, self._lattice.mu_v_ap,
            self._lattice.v.beta_corr, self._lattice.v.mu_corr,
        )

    def get_machine_info(self) -> dict[str, Any]:
        """Return NOMINAL machine parameters only.

        No true offsets, gains, or perturbed optics are returned.
        """
        lat = self._lattice
        return {
            "h": {
                "n_corr": lat.h.n_corr,
                "n_sem": lat.h.n_sem,
                "s_corr": lat.h.s_corr.tolist(),
                "beta_corr": lat.h.beta_corr.tolist(),
                "alpha_corr": lat.h.alpha_corr.tolist(),
                "mu_corr": lat.h.mu_corr.tolist(),
                "s_sem": lat.h.s_sem.tolist(),
                "beta_sem": lat.h.beta_sem.tolist(),
                "alpha_sem": lat.h.alpha_sem.tolist(),
                "mu_sem": lat.h.mu_sem.tolist(),
                "aperture_sem": lat.h.aperture_sem.tolist(),
                "sigma_sem": lat.h.sigma_sem.tolist(),
                "foil_length": lat.h.foil_length.tolist(),
                "i_max": lat.h.i_max.tolist(),
            },
            "v": {
                "n_corr": lat.v.n_corr,
                "n_sem": lat.v.n_sem,
                "s_corr": lat.v.s_corr.tolist(),
                "beta_corr": lat.v.beta_corr.tolist(),
                "alpha_corr": lat.v.alpha_corr.tolist(),
                "mu_corr": lat.v.mu_corr.tolist(),
                "s_sem": lat.v.s_sem.tolist(),
                "beta_sem": lat.v.beta_sem.tolist(),
                "alpha_sem": lat.v.alpha_sem.tolist(),
                "mu_sem": lat.v.mu_sem.tolist(),
                "aperture_sem": lat.v.aperture_sem.tolist(),
                "sigma_sem": lat.v.sigma_sem.tolist(),
                "foil_length": lat.v.foil_length.tolist(),
                "i_max": lat.v.i_max.tolist(),
            },
            "checkpoints": {
                "s_ap": lat.s_ap.tolist(),
                "a_h_ap": lat.a_h_ap.tolist(),
                "a_v_ap": lat.a_v_ap.tolist(),
                "beta_h_ap": lat.beta_h_ap.tolist(),
                "alpha_h_ap": lat.alpha_h_ap.tolist(),
                "mu_h_ap": lat.mu_h_ap.tolist(),
                "beta_v_ap": lat.beta_v_ap.tolist(),
                "alpha_v_ap": lat.alpha_v_ap.tolist(),
                "mu_v_ap": lat.mu_v_ap.tolist(),
                "sem_h_indices_in_ap": lat.sem_h_indices_in_ap.tolist(),
                "sem_v_indices_in_ap": lat.sem_v_indices_in_ap.tolist(),
                "corr_h_indices_in_ap": lat.corr_h_indices_in_ap.tolist(),
                "corr_v_indices_in_ap": lat.corr_v_indices_in_ap.tolist(),
            },
            "r_sem_h": self._nominal_r_sem_h.tolist(),
            "r_ap_h": self._nominal_r_ap_h.tolist(),
            "r_sem_v": self._nominal_r_sem_v.tolist(),
            "r_ap_v": self._nominal_r_ap_v.tolist(),
            "nu": 0.25,
            "theta_max": float(self._theta_max),
            "budget": self._budget,
            "shots_used": self._sim.shots_used,
            "shots_remaining": max(0, self._budget - self._sim.shots_used),
            "submitted": self._submitted,
        }


    def set_correctors(
        self,
        h: list[float] | np.ndarray,
        v: list[float] | np.ndarray,
    ) -> dict[str, Any]:
        """Command corrector angles. Costs 1 shot. Returns noisy readings."""
        if self._submitted:
            raise RuntimeError("Episode already submitted. No further commands allowed.")
        if self._sim.shots_used >= self._budget:
            raise RuntimeError(f"Shot budget ({self._budget}) exhausted.")

        arr_h = np.asarray(h, dtype=np.float64)
        arr_v = np.asarray(v, dtype=np.float64)

        if len(arr_h) != self._lattice.h.n_corr:
            raise ValueError(f"Expected {self._lattice.h.n_corr} horizontal angles, got {len(arr_h)}")
        if len(arr_v) != self._lattice.v.n_corr:
            raise ValueError(f"Expected {self._lattice.v.n_corr} vertical angles, got {len(arr_v)}")

        if np.any(np.isnan(arr_h)) or np.any(np.isnan(arr_v)) or np.any(np.isinf(arr_h)) or np.any(np.isinf(arr_v)):
            raise ValueError("Corrector commands contain NaN or Inf")

        arr_h = np.clip(arr_h, -self._theta_max, self._theta_max)
        arr_v = np.clip(arr_v, -self._theta_max, self._theta_max)

        step_res = self._sim.set_correctors(arr_h, arr_v, clip=False)

        obs_h = [{"ratio": o.ratio, "level": o.level} for o in step_res.obs_h]
        obs_v = [{"ratio": o.ratio, "level": o.level} for o in step_res.obs_v]

        entry = {
            "shot": self._sim.shots_used,
            "theta_h": arr_h.tolist(),
            "theta_v": arr_v.tolist(),
            "obs_h": obs_h,
            "obs_v": obs_v,
        }
        self._history.append(entry)

        return {
            "h": obs_h,
            "v": obs_v,
            "shots_used": self._sim.shots_used,
            "shots_remaining": max(0, self._budget - self._sim.shots_used),
        }

    def get_history(self) -> list[dict[str, Any]]:
        """Return history of settings and readings."""
        return [dict(e) for e in self._history]

    def submit(self) -> dict[str, Any]:
        """Freeze the episode. Further set_correctors calls will be rejected."""
        self._submitted = True
        return {
            "submitted": True,
            "shots_used": self._sim.shots_used,
            "shots_remaining": max(0, self._budget - self._sim.shots_used),
        }


def evaluate_env(env: BeamlineEnv) -> ScoreDetails:
    """Evaluator-only function: computes true noise-free score from internal simulator."""
    sim = env._sim
    s0 = sim.initial_result.s_mean
    sf = sim.last_result.s_mean

    return compute_reward(
        s0=s0,
        sf=sf,
        shots_used=sim.shots_used,
        budget=env._budget,
        loss_s=sim.last_result.loss_s,
        is_lost=sim.last_result.is_lost,
        s_hat_h=sim.last_result.s_hat_h,
        s_hat_v=sim.last_result.s_hat_v,
    )


def get_env_true_trajectory(env: BeamlineEnv) -> dict[str, Any]:
    """Evaluator-only function: returns true trajectory and loss info for replay logging."""
    sim = env._sim
    res = sim.last_result
    return {
        "x_sem_h": res.x_sem_h.tolist(),
        "x_sem_v": res.x_sem_v.tolist(),
        "is_lost": res.is_lost,
        "loss_s": res.loss_s,
        "first_lost_sem_h": res.first_lost_sem_h,
        "first_lost_sem_v": res.first_lost_sem_v,
        "s_mean": res.s_mean,
        "s_hat_h": res.s_hat_h.tolist(),
        "s_hat_v": res.s_hat_v.tolist(),
        "distance_d": res.distance_d,
    }

