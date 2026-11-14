"""Physics simulation of the CERN TT20 transfer line (Adam Štafa report §1.1).

Implements equations (1)-(7), (10)-(11):
- Response matrix trajectory displacement: x = R(Delta_theta - g * theta)
- Joint two-plane beam loss on common aperture checkpoints
- Noisy SEM foil observations (ratio, level) with uniform background noise
- Noise-free true symmetry (S, S_hat, S_mean) for verification & scoring
- Distance metric d = sqrt(mean((x / sigma)^2))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
from scipy.special import ndtr

from beamsim.lattice import Lattice, load_lattice
from beamsim.randomize import PerturbedModel, create_perturbed_model


@dataclass
class SEMObservation:
    """Noisy observation for a single SEM monitor."""
    ratio: float  # I_tilde_1 / (I_tilde_1 + I_tilde_2)
    level: float  # (I_tilde_1 + I_tilde_2) / I_max


@dataclass
class StepResult:
    """Full outcome of a corrector setting command."""
    obs_h: list[SEMObservation]
    obs_v: list[SEMObservation]
    s_mean: float               # True noise-free mean S_hat over all 21 monitors
    s_hat_h: np.ndarray         # (11,) True noise-free S_hat for H monitors
    s_hat_v: np.ndarray         # (10,) True noise-free S_hat for V monitors
    s_raw_h: np.ndarray         # (11,) True S (eq. 5)
    s_raw_v: np.ndarray         # (10,) True S (eq. 5)
    x_sem_h: np.ndarray         # (11,) True beam positions at H SEMs
    x_sem_v: np.ndarray         # (10,) True beam positions at V SEMs
    is_lost: bool               # True if beam hit aperture anywhere
    loss_s: float | None        # Longitudinal position s where beam was lost
    first_lost_sem_h: int | None # Index of first H SEM reading background only
    first_lost_sem_v: int | None # Index of first V SEM reading background only
    distance_d: float           # eq. (7) normalized distance across all SEMs


class BeamlineSim:
    """Simulator for the TT20 beamline with hidden offsets and model mismatch."""

    def __init__(
        self,
        lattice: Lattice | None = None,
        model: PerturbedModel | None = None,
        seed: int = 42,
        spread: str = "medium",
        delta_theta_h: np.ndarray | None = None,
        delta_theta_v: np.ndarray | None = None,
        theta_max: float | None = None,
        nu: float | None = None,
    ) -> None:
        self.lattice = lattice if lattice is not None else load_lattice()
        self.theta_max = theta_max if theta_max is not None else self.lattice.theta_max
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Build model (perturbed or nominal)
        if model is not None:
            self.model = model
        else:
            # Table 1 randomization
            self.model = create_perturbed_model(self.lattice, seed=seed, spread=spread)

        if nu is not None:
            self.model.nu = nu

        # Sample hidden offsets Delta_theta ~ U[-theta_max, theta_max] if not provided
        if delta_theta_h is not None:
            self.delta_theta_h = np.array(delta_theta_h, dtype=np.float64).copy()
        else:
            self.delta_theta_h = self.rng.uniform(-self.theta_max, self.theta_max, size=self.lattice.h.n_corr)

        if delta_theta_v is not None:
            self.delta_theta_v = np.array(delta_theta_v, dtype=np.float64).copy()
        else:
            self.delta_theta_v = self.rng.uniform(-self.theta_max, self.theta_max, size=self.lattice.v.n_corr)

        # Track history
        self.shots_used = 0
        self.current_theta_h = np.zeros(self.lattice.h.n_corr, dtype=np.float64)
        self.current_theta_v = np.zeros(self.lattice.v.n_corr, dtype=np.float64)
        self.initial_result: StepResult | None = None
        self.last_result: StepResult | None = None

        # Run initial state (theta = 0, costs 0 shots)
        self.initial_result = self._evaluate(self.current_theta_h, self.current_theta_v, add_noise=True)
        self.last_result = self.initial_result

    def set_correctors(
        self,
        theta_h: list[float] | np.ndarray,
        theta_v: list[float] | np.ndarray,
        clip: bool = True,
    ) -> StepResult:
        """Command absolute corrector angles, increment shot counter, return readings."""
        th_h = np.array(theta_h, dtype=np.float64)
        th_v = np.array(theta_v, dtype=np.float64)

        if np.any(np.isnan(th_h)) or np.any(np.isnan(th_v)) or np.any(np.isinf(th_h)) or np.any(np.isinf(th_v)):
            raise ValueError("Corrector angles contain NaN or Inf values")

        if clip:
            th_h = np.clip(th_h, -self.theta_max, self.theta_max)
            th_v = np.clip(th_v, -self.theta_max, self.theta_max)

        self.shots_used += 1
        self.current_theta_h = th_h
        self.current_theta_v = th_v

        result = self._evaluate(th_h, th_v, add_noise=True)
        self.last_result = result
        return result

    def _evaluate(
        self,
        theta_h: np.ndarray,
        theta_v: np.ndarray,
        add_noise: bool = True,
    ) -> StepResult:
        """Calculate beam trajectory, loss, foil intensities and metrics."""
        # Corrector gains: x = R(Delta_theta - g * theta)
        kick_h = self.delta_theta_h - self.model.h.corrector_gains * theta_h
        kick_v = self.delta_theta_v - self.model.v.corrector_gains * theta_v

        # Trajectory at all aperture checkpoints: x_ap = R_ap @ kick (eq. 1, 10)
        x_ap_h = self.model.h.r_matrix_ap @ kick_h
        x_ap_v = self.model.v.r_matrix_ap @ kick_v

        # Positions at SEM monitors
        x_sem_h = self.model.h.r_matrix_sem @ kick_h
        x_sem_v = self.model.v.r_matrix_sem @ kick_v

        # Joint collision detection (report §1.1.1):
        # Beam is lost at the first checkpoint where |x_h| > a_h or |x_v| > a_v
        loss_mask = (np.abs(x_ap_h) > self.lattice.a_h_ap) | (np.abs(x_ap_v) > self.lattice.a_v_ap)
        is_lost = bool(np.any(loss_mask))

        if is_lost:
            first_loss_idx = int(np.where(loss_mask)[0][0])
            loss_s = float(self.lattice.s_ap[first_loss_idx])
        else:
            loss_s = None

        # Determine which SEMs read background only
        lost_sem_h = np.zeros(self.lattice.h.n_sem, dtype=bool) if loss_s is None else (self.lattice.h.s_sem >= loss_s)
        lost_sem_v = np.zeros(self.lattice.v.n_sem, dtype=bool) if loss_s is None else (self.lattice.v.s_sem >= loss_s)

        first_lost_sem_h = int(np.where(lost_sem_h)[0][0]) if np.any(lost_sem_h) else None
        first_lost_sem_v = int(np.where(lost_sem_v)[0][0]) if np.any(lost_sem_v) else None

        # Compute foil readings for both planes (eq. 2-6)
        obs_h, s_hat_h, s_raw_h = self._compute_plane_foils(
            x_sem=x_sem_h,
            lost_mask=lost_sem_h,
            plane_model=self.model.h,
            add_noise=add_noise,
        )

        obs_v, s_hat_v, s_raw_v = self._compute_plane_foils(
            x_sem=x_sem_v,
            lost_mask=lost_sem_v,
            plane_model=self.model.v,
            add_noise=add_noise,
        )

        # True S_mean over all 21 monitors (report §1.1.2)
        all_s_hat = np.concatenate([s_hat_h, s_hat_v])
        s_mean = float(np.mean(all_s_hat))

        # Normalized distance d = sqrt(mean((x / sigma)^2)) (eq. 7)
        dist_terms = np.concatenate([
            (x_sem_h / self.model.h.sigma_sem)**2,
            (x_sem_v / self.model.v.sigma_sem)**2,
        ])
        distance_d = float(np.sqrt(np.mean(dist_terms)))

        return StepResult(
            obs_h=obs_h,
            obs_v=obs_v,
            s_mean=s_mean,
            s_hat_h=s_hat_h,
            s_hat_v=s_hat_v,
            s_raw_h=s_raw_h,
            s_raw_v=s_raw_v,
            x_sem_h=x_sem_h,
            x_sem_v=x_sem_v,
            is_lost=is_lost,
            loss_s=loss_s,
            first_lost_sem_h=first_lost_sem_h,
            first_lost_sem_v=first_lost_sem_v,
            distance_d=distance_d,
        )

    def _compute_plane_foils(
        self,
        x_sem: np.ndarray,
        lost_mask: np.ndarray,
        plane_model: Any,
        add_noise: bool,
    ) -> tuple[list[SEMObservation], np.ndarray, np.ndarray]:
        """Compute true and noisy foil readings for monitors in one plane."""
        n_sem = len(x_sem)
        obs_list: list[SEMObservation] = []
        s_hat = np.zeros(n_sem, dtype=np.float64)
        s_raw = np.zeros(n_sem, dtype=np.float64)

        for i in range(n_sem):
            sigma = plane_model.sigma_sem[i]
            l_foil = plane_model.foil_length[i]
            i_max = plane_model.i_max[i]

            if lost_mask[i]:
                # Beam lost before/at this monitor: true intensity is 0 (reads background only)
                i1_true = 0.0
                i2_true = 0.0
                s_raw[i] = 0.0
                s_hat[i] = 0.0
            else:
                # Intercepted fractions for centered/displaced beam (eq. 2)
                x = x_sem[i]
                i1_true = float(ndtr(-x / sigma) - ndtr((-l_foil - x) / sigma))
                i2_true = float(ndtr((l_foil - x) / sigma) - ndtr(-x / sigma))

                # Symmetry (eq. 5) & Normalized symmetry (eq. 6)
                total_i = i1_true + i2_true
                if total_i > 1e-12:
                    diff_i = abs(i1_true - i2_true)
                    s_val = np.sqrt(max(0.0, 1.0 - diff_i / total_i))
                    s_raw[i] = float(s_val)
                    s_hat[i] = float(s_val * total_i / i_max)
                else:
                    s_raw[i] = 0.0
                    s_hat[i] = 0.0

            # Noise: epsilon ~ U[0, nu] (eq. 3)
            if add_noise:
                eps1 = self.rng.uniform(0.0, self.model.nu)
                eps2 = self.rng.uniform(0.0, self.model.nu)
            else:
                eps1 = 0.0
                eps2 = 0.0

            i1_meas = i1_true + eps1
            i2_meas = i2_true + eps2
            denom = i1_meas + i2_meas

            # Reported observations (eq. 4, 11)
            ratio = float(i1_meas / denom) if denom > 1e-12 else 0.5
            level = float(denom / i_max)

            obs_list.append(SEMObservation(ratio=ratio, level=level))

        return obs_list, s_hat, s_raw


    s_raw_v: np.ndarray         # (10,) True S (eq. 5)
    x_sem_h: np.ndarray         # (11,) True beam positions at H SEMs
    x_sem_v: np.ndarray         # (10,) True beam positions at V SEMs
    is_lost: bool               # True if beam hit aperture anywhere
    loss_s: float | None        # Longitudinal position s where beam was lost
    first_lost_sem_h: int | None # Index of first H SEM reading background only
    first_lost_sem_v: int | None # Index of first V SEM reading background only
    distance_d: float           # eq. (7) normalized distance across all SEMs
