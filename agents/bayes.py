"""Bayesian particle filter reference agent for TT20 transfer line (Adam Štafa report §3).

Features:
- Joint 22-dimensional particle filter across both planes (10 H + 12 V)
- Uniform prior on [-theta_max, theta_max]^22
- Approximate Huber likelihood (eq. 14-15) with ratio term dropped where loss predicted
- Multinomial resampling with roughening 0.5 (eq. 16)
- Action selection rule with alpha = 0.1, beta = 0.6 (eq. 17)
- Stops and submits when predicted S_mean under posterior mean >= 0.97 or budget exhausted
- Chunked particle evaluation keeping memory usage under ~100 MB
"""

from __future__ import annotations

import time
from typing import Any
import numpy as np
from scipy.special import logsumexp, ndtr

from beamsim.env_api import BeamlineEnv


def huber_loss(z: np.ndarray) -> np.ndarray:
    """Standard Huber loss: 0.5*z^2 if |z| <= 1 else |z| - 0.5 (eq. 15)."""
    abs_z = np.abs(z)
    mask = abs_z <= 1.0
    return np.where(mask, 0.5 * z**2, abs_z - 0.5)


class BayesAgent:
    """Bayesian particle filter reference agent using BeamlineEnv."""

    def __init__(
        self,
        n_particles: int = 50_000,
        chunk_size: int = 10_000,
        alpha: float = 0.1,
        beta: float = 0.6,
        target_s_mean: float = 0.97,
        roughening: float = 0.5,
        seed: int = 0,
    ) -> None:
        self.n_particles = int(n_particles)
        self.chunk_size = int(chunk_size)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.target_s_mean = float(target_s_mean)
        self.roughening = float(roughening)
        self.rng = np.random.default_rng(seed)

    def solve(
        self,
        env: BeamlineEnv,
        on_shot_callback: Any = None,
    ) -> list[dict[str, Any]]:
        """Run the Bayes agent loop on the environment until submit or budget exhaustion."""
        info = env.get_machine_info()

        theta_max = np.float32(info["theta_max"])
        nu = np.float32(info["nu"])
        half_nu = np.float32(0.5 * nu)
        inv_nu = np.float32(1.0 / nu)
        inv_2nu = np.float32(1.0 / (2.0 * nu))
        budget = info["budget"]

        # Precompute coarse aperture checkpoints (~55-62 points)
        s_ap_all = np.array(info["checkpoints"]["s_ap"], dtype=np.float32)
        elem_indices = set(
            info["checkpoints"]["sem_h_indices_in_ap"]
            + info["checkpoints"]["sem_v_indices_in_ap"]
            + info["checkpoints"]["corr_h_indices_in_ap"]
            + info["checkpoints"]["corr_v_indices_in_ap"]
        )
        for k, s in enumerate(s_ap_all):
            if abs(s % 20.0) < 2.6:
                elem_indices.add(k)
        coarse_idx = np.array(sorted(elem_indices), dtype=np.int64)

        r_ap_coarse_h_T = np.array(info["r_ap_h"], dtype=np.float32).T[:, coarse_idx]
        r_ap_coarse_v_T = np.array(info["r_ap_v"], dtype=np.float32).T[:, coarse_idx]
        a_h_coarse = np.array(info["checkpoints"]["a_h_ap"], dtype=np.float32)[coarse_idx]
        a_v_coarse = np.array(info["checkpoints"]["a_v_ap"], dtype=np.float32)[coarse_idx]

        sem_h_coarse = np.array([np.where(coarse_idx == i)[0][0] for i in info["checkpoints"]["sem_h_indices_in_ap"]], dtype=np.int64)
        sem_v_coarse = np.array([np.where(coarse_idx == i)[0][0] for i in info["checkpoints"]["sem_v_indices_in_ap"]], dtype=np.int64)

        r_sem_h_T = np.array(info["r_sem_h"], dtype=np.float32).T
        r_sem_v_T = np.array(info["r_sem_v"], dtype=np.float32).T

        sigma_h = np.array(info["h"]["sigma_sem"], dtype=np.float32)
        sigma_v = np.array(info["v"]["sigma_sem"], dtype=np.float32)
        foil_l_h = np.array(info["h"]["foil_length"], dtype=np.float32)
        foil_l_v = np.array(info["v"]["foil_length"], dtype=np.float32)
        i_max_h = np.array(info["h"]["i_max"], dtype=np.float32)
        i_max_v = np.array(info["v"]["i_max"], dtype=np.float32)

        particles = self.rng.uniform(-theta_max, theta_max, size=(self.n_particles, 22)).astype(np.float32)
        agent_logs: list[dict[str, Any]] = []

        for shot_idx in range(1, budget + 1):
            mean_delta = np.mean(particles, axis=0)

            pred_s_mean, p_lost = self._evaluate_particles_s_mean(
                particles=particles,
                theta=mean_delta,
                r_sem_h_T=r_sem_h_T,
                r_ap_h_T=r_ap_coarse_h_T,
                r_sem_v_T=r_sem_v_T,
                r_ap_v_T=r_ap_coarse_v_T,
                a_h_ap=a_h_coarse,
                a_v_ap=a_v_coarse,
                sem_h_indices=sem_h_coarse,
                sem_v_indices=sem_v_coarse,
                sigma_h=sigma_h,
                sigma_v=sigma_v,
                foil_l_h=foil_l_h,
                foil_l_v=foil_l_v,
                i_max_h=i_max_h,
                i_max_v=i_max_v,
            )

            if on_shot_callback is not None:
                on_shot_callback(shot_idx, mean_delta, p_lost)


            is_confident = (pred_s_mean >= self.target_s_mean)
            is_last_shot = (shot_idx == budget)

            if is_confident or is_last_shot:
                action = np.clip(mean_delta, -theta_max, theta_max)
                act_h = action[:10]
                act_v = action[10:]
                obs = env.set_correctors(act_h, act_v)

                shot_log = {
                    "shot": shot_idx,
                    "theta_h": act_h.tolist(),
                    "theta_v": act_v.tolist(),
                    "obs_h": obs["h"],
                    "obs_v": obs["v"],
                    "posterior_mean_h": mean_delta[:10].tolist(),
                    "posterior_mean_v": mean_delta[10:].tolist(),
                    "p_lost": float(p_lost),
                    "predicted_s_mean": float(pred_s_mean),
                    "action_type": "submit_mean" if is_confident else "last_shot_mean",
                }
                agent_logs.append(shot_log)
                env.submit()
                break

            idx_sample = self.rng.integers(0, self.n_particles)
            sample_delta = particles[idx_sample]
            explore_factor = self.alpha + self.beta * p_lost
            command_theta = mean_delta + explore_factor * (sample_delta - mean_delta)
            command_theta = np.clip(command_theta, -theta_max, theta_max)

            act_h = command_theta[:10]
            act_v = command_theta[10:]

            obs = env.set_correctors(act_h, act_v)

            shot_log = {
                "shot": shot_idx,
                "theta_h": act_h.tolist(),
                "theta_v": act_v.tolist(),
                "obs_h": obs["h"],
                "obs_v": obs["v"],
                "posterior_mean_h": mean_delta[:10].tolist(),
                "posterior_mean_v": mean_delta[10:].tolist(),
                "p_lost": float(p_lost),
                "predicted_s_mean": float(pred_s_mean),
                "action_type": "explore",
            }
            agent_logs.append(shot_log)

            r_obs_h = np.array([o["ratio"] for o in obs["h"]], dtype=np.float32)
            l_obs_h = np.array([o["level"] for o in obs["h"]], dtype=np.float32)
            r_obs_v = np.array([o["ratio"] for o in obs["v"]], dtype=np.float32)
            l_obs_v = np.array([o["level"] for o in obs["v"]], dtype=np.float32)

            log_weights = self._compute_log_likelihood(
                particles=particles,
                theta=command_theta,
                r_obs_h=r_obs_h,
                l_obs_h=l_obs_h,
                r_obs_v=r_obs_v,
                l_obs_v=l_obs_v,
                r_sem_h_T=r_sem_h_T,
                r_ap_h_T=r_ap_coarse_h_T,
                r_sem_v_T=r_sem_v_T,
                r_ap_v_T=r_ap_coarse_v_T,
                a_h_ap=a_h_coarse,
                a_v_ap=a_v_coarse,
                sem_h_indices=sem_h_coarse,
                sem_v_indices=sem_v_coarse,
                sigma_h=sigma_h,
                sigma_v=sigma_v,
                foil_l_h=foil_l_h,
                foil_l_v=foil_l_v,
                i_max_h=i_max_h,
                i_max_v=i_max_v,
                nu=nu,
                half_nu=half_nu,
                inv_nu=inv_nu,
                inv_2nu=inv_2nu,
            )

            normalized_log_w = log_weights - logsumexp(log_weights)
            weights = np.exp(normalized_log_w)
            weights /= np.sum(weights)

            resampled_indices = self.rng.choice(self.n_particles, size=self.n_particles, p=weights)
            particles = particles[resampled_indices]

            cov = np.cov(particles, rowvar=False).astype(np.float32)
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            eigenvals = np.maximum(eigenvals, 1e-14, dtype=np.float32)
            sqrt_cov = (eigenvecs @ np.diag(np.sqrt(eigenvals)) @ eigenvecs.T).astype(np.float32)

            jitter = self.roughening * (self.rng.normal(size=(self.n_particles, 22)).astype(np.float32) @ sqrt_cov)
            particles = np.clip(particles + jitter, -theta_max, theta_max)

        return agent_logs



    def _compute_log_likelihood(
        self,
        particles: np.ndarray,
        theta: np.ndarray,
        r_obs_h: np.ndarray,
        l_obs_h: np.ndarray,
        r_obs_v: np.ndarray,
        l_obs_v: np.ndarray,
        r_sem_h_T: np.ndarray,
        r_ap_h_T: np.ndarray,
        r_sem_v_T: np.ndarray,
        r_ap_v_T: np.ndarray,
        a_h_ap: np.ndarray,
        a_v_ap: np.ndarray,
        sem_h_indices: np.ndarray,
        sem_v_indices: np.ndarray,
        sigma_h: np.ndarray,
        sigma_v: np.ndarray,
        foil_l_h: np.ndarray,
        foil_l_v: np.ndarray,
        i_max_h: np.ndarray,
        i_max_v: np.ndarray,
        nu: np.float32,
        half_nu: np.float32,
        inv_nu: np.float32,
        inv_2nu: np.float32,
    ) -> np.ndarray:
        """Compute Huber approximate log-likelihood for all particles in chunks (eq. 14-15)."""
        n_p = particles.shape[0]
        log_weights = np.zeros(n_p, dtype=np.float32)

        for start_idx in range(0, n_p, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, n_p)
            chunk = particles[start_idx:end_idx]

            kick_h = chunk[:, :10] - theta[:10]
            kick_v = chunk[:, 10:] - theta[10:]

            x_ap_h = kick_h @ r_ap_h_T
            x_ap_v = kick_v @ r_ap_v_T

            loss_mask = (np.abs(x_ap_h) > a_h_ap) | (np.abs(x_ap_v) > a_v_ap)
            cum_lost = np.maximum.accumulate(loss_mask, axis=1)

            sem_lost_h = cum_lost[:, sem_h_indices]
            sem_lost_v = cum_lost[:, sem_v_indices]

            x_sem_h = kick_h @ r_sem_h_T
            x_sem_v = kick_v @ r_sem_v_T

            phi_neg_h = ndtr(-x_sem_h / sigma_h)
            i1_h = phi_neg_h - ndtr((-foil_l_h - x_sem_h) / sigma_h)
            i2_h = ndtr((foil_l_h - x_sem_h) / sigma_h) - phi_neg_h
            i1_h = np.where(sem_lost_h, 0.0, i1_h)
            i2_h = np.where(sem_lost_h, 0.0, i2_h)

            phi_neg_v = ndtr(-x_sem_v / sigma_v)
            i1_v = phi_neg_v - ndtr((-foil_l_v - x_sem_v) / sigma_v)
            i2_v = ndtr((foil_l_v - x_sem_v) / sigma_v) - phi_neg_v
            i1_v = np.where(sem_lost_v, 0.0, i1_v)
            i2_v = np.where(sem_lost_v, 0.0, i2_v)

            i1_tilde_h = i1_h + half_nu
            t_tilde_h = i1_h + i2_h + nu

            i1_tilde_v = i1_v + half_nu
            t_tilde_v = i1_v + i2_v + nu

            zr_h = (r_obs_h * t_tilde_h - i1_tilde_h) * inv_nu
            zl_h = (l_obs_h * i_max_h - t_tilde_h) * inv_2nu

            zr_v = (r_obs_v * t_tilde_v - i1_tilde_v) * inv_nu
            zl_v = (l_obs_v * i_max_v - t_tilde_v) * inv_2nu


            h_loss_r_h = np.where(sem_lost_h, 0.0, huber_loss(zr_h))
            h_loss_l_h = huber_loss(zl_h)

            h_loss_r_v = np.where(sem_lost_v, 0.0, huber_loss(zr_v))
            h_loss_l_v = huber_loss(zl_v)

            total_huber = (
                np.sum(h_loss_r_h + h_loss_l_h, axis=1)
                + np.sum(h_loss_r_v + h_loss_l_v, axis=1)
            )

            log_weights[start_idx:end_idx] = -total_huber

        return log_weights



    def _evaluate_particles_s_mean(
        self,
        particles: np.ndarray,
        theta: np.ndarray,
        r_sem_h_T: np.ndarray,
        r_ap_h_T: np.ndarray,
        r_sem_v_T: np.ndarray,
        r_ap_v_T: np.ndarray,
        a_h_ap: np.ndarray,
        a_v_ap: np.ndarray,
        sem_h_indices: np.ndarray,
        sem_v_indices: np.ndarray,
        sigma_h: np.ndarray,
        sigma_v: np.ndarray,
        foil_l_h: np.ndarray,
        foil_l_v: np.ndarray,
        i_max_h: np.ndarray,
        i_max_v: np.ndarray,
    ) -> tuple[float, float]:
        """Compute mean predicted S_mean and fraction of particles lost under commanded setting theta."""
        if particles.shape[0] > 5000:
            step = max(1, particles.shape[0] // 5000)
            eval_p = particles[::step]
        else:
            eval_p = particles

        n_p = eval_p.shape[0]
        s_mean_total = 0.0
        lost_count = 0

        for start_idx in range(0, n_p, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, n_p)
            chunk = eval_p[start_idx:end_idx]

            kick_h = chunk[:, :10] - theta[:10]
            kick_v = chunk[:, 10:] - theta[10:]

            x_ap_h = kick_h @ r_ap_h_T
            x_ap_v = kick_v @ r_ap_v_T

            loss_mask = (np.abs(x_ap_h) > a_h_ap) | (np.abs(x_ap_v) > a_v_ap)
            cum_lost = np.maximum.accumulate(loss_mask, axis=1)

            lost_line = cum_lost[:, -1]
            lost_count += int(np.sum(lost_line))

            sem_lost_h = cum_lost[:, sem_h_indices]
            sem_lost_v = cum_lost[:, sem_v_indices]

            x_sem_h = kick_h @ r_sem_h_T
            x_sem_v = kick_v @ r_sem_v_T

            phi_neg_h = ndtr(-x_sem_h / sigma_h)
            i1_h = phi_neg_h - ndtr((-foil_l_h - x_sem_h) / sigma_h)
            i2_h = ndtr((foil_l_h - x_sem_h) / sigma_h) - phi_neg_h
            tot_h = i1_h + i2_h
            diff_h = np.abs(i1_h - i2_h)
            s_raw_h = np.sqrt(np.maximum(0.0, 1.0 - diff_h / (tot_h + 1e-12)))
            s_hat_h = s_raw_h * (tot_h / i_max_h)
            s_hat_h = np.where(sem_lost_h, 0.0, s_hat_h)

            phi_neg_v = ndtr(-x_sem_v / sigma_v)
            i1_v = phi_neg_v - ndtr((-foil_l_v - x_sem_v) / sigma_v)
            i2_v = ndtr((foil_l_v - x_sem_v) / sigma_v) - phi_neg_v
            tot_v = i1_v + i2_v
            diff_v = np.abs(i1_v - i2_v)
            s_raw_v = np.sqrt(np.maximum(0.0, 1.0 - diff_v / (tot_v + 1e-12)))
            s_hat_v = s_raw_v * (tot_v / i_max_v)
            s_hat_v = np.where(sem_lost_v, 0.0, s_hat_v)

            s_mean_chunk = np.mean(np.concatenate([s_hat_h, s_hat_v], axis=1), axis=1)
            s_mean_total += float(np.sum(s_mean_chunk))

        pred_s_mean = float(s_mean_total / n_p)
        p_lost = float(lost_count / n_p)
        return pred_s_mean, p_lost


