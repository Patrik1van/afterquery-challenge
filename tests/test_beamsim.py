"""Pytest suite for TT20 beamline simulator (beamsim).

Verifies the 4 core cases:
1. A centred beam scores about 1 (s_mean ~ 1.0).
2. theta = Delta_theta is solved (nominal), and theta = Delta_theta / g is solved (perturbed).
3. A huge theta loses the beam at once (at monitor 0).
4. The same seed gives identical readings.
"""

import numpy as np
import pytest

from beamsim.lattice import build_synthetic_lattice
from beamsim.scoring import compute_reward
from beamsim.sim import BeamlineSim


def test_centred_beam_scores_about_one():
    """A centered beam with zero offsets and zero angles scores about 1."""
    lattice = build_synthetic_lattice()
    zero_h = np.zeros(lattice.h.n_corr)
    zero_v = np.zeros(lattice.v.n_corr)

    sim = BeamlineSim(
        lattice=lattice,
        seed=42,
        spread="off",
        delta_theta_h=zero_h,
        delta_theta_v=zero_v,
    )

    res = sim.set_correctors(zero_h, zero_v)

    assert not res.is_lost
    assert res.loss_s is None
    assert res.first_lost_sem_h is None
    assert res.first_lost_sem_v is None
    np.testing.assert_allclose(res.x_sem_h, 0.0, atol=1e-12)
    np.testing.assert_allclose(res.x_sem_v, 0.0, atol=1e-12)
    assert pytest.approx(res.s_mean, abs=1e-6) == 1.0

    score = compute_reward(s0=0.0, sf=res.s_mean, shots_used=1, budget=40)
    assert score.solved is True
    assert score.reward >= 0.99


def test_theta_equals_delta_theta_is_solved_nominal():
    """On unperturbed model, commanding theta = Delta_theta perfectly centers beam and solves."""
    lattice = build_synthetic_lattice()
    sim = BeamlineSim(lattice=lattice, seed=123, spread="off")

    # Command theta = Delta_theta
    res = sim.set_correctors(sim.delta_theta_h, sim.delta_theta_v)

    assert not res.is_lost
    np.testing.assert_allclose(res.x_sem_h, 0.0, atol=1e-12)
    np.testing.assert_allclose(res.x_sem_v, 0.0, atol=1e-12)
    assert res.s_mean >= 0.95
    assert pytest.approx(res.s_mean, abs=1e-6) == 1.0

    score = compute_reward(s0=sim.initial_result.s_mean, sf=res.s_mean, shots_used=sim.shots_used, budget=40)
    assert score.solved is True


def test_theta_equals_delta_theta_over_g_is_solved_perturbed():
    """Under domain randomization, commanding theta = Delta_theta / g solves the line."""
    lattice = build_synthetic_lattice()
    sim = BeamlineSim(lattice=lattice, seed=456, spread="medium")

    # Command theta = Delta_theta / g
    command_h = sim.delta_theta_h / sim.model.h.corrector_gains
    command_v = sim.delta_theta_v / sim.model.v.corrector_gains

    res = sim.set_correctors(command_h, command_v)

    assert not res.is_lost
    np.testing.assert_allclose(res.x_sem_h, 0.0, atol=1e-12)
    np.testing.assert_allclose(res.x_sem_v, 0.0, atol=1e-12)
    assert res.s_mean >= 0.95
    assert pytest.approx(res.s_mean, abs=1e-6) == 1.0

    score = compute_reward(s0=sim.initial_result.s_mean, sf=res.s_mean, shots_used=sim.shots_used, budget=40)
    assert score.solved is True


def test_huge_theta_loses_beam_at_once():
    """A huge commanded angle loses the beam at once before or at monitor 0."""
    lattice = build_synthetic_lattice()
    sim = BeamlineSim(lattice=lattice, seed=789, spread="off")

    huge_h = np.ones(lattice.h.n_corr) * 0.1  # 100 mrad >> theta_max
    huge_v = np.ones(lattice.v.n_corr) * 0.1

    # Call with clip=False to simulate huge kick
    res = sim.set_correctors(huge_h, huge_v, clip=False)

    assert res.is_lost is True
    assert res.loss_s is not None
    assert res.loss_s <= min(lattice.h.s_sem[0], lattice.v.s_sem[0])
    assert res.first_lost_sem_h == 0
    assert res.first_lost_sem_v == 0
    np.testing.assert_allclose(res.s_hat_h, 0.0)
    np.testing.assert_allclose(res.s_hat_v, 0.0)
    assert res.s_mean == 0.0


def test_same_seed_gives_identical_readings():
    """Same seed and actions yield identical noisy readings and scores."""
    lattice = build_synthetic_lattice()

    sim1 = BeamlineSim(lattice=lattice, seed=999, spread="medium")
    sim2 = BeamlineSim(lattice=lattice, seed=999, spread="medium")

    # Initial offsets & gains match
    np.testing.assert_allclose(sim1.delta_theta_h, sim2.delta_theta_h)
    np.testing.assert_allclose(sim1.delta_theta_v, sim2.delta_theta_v)
    np.testing.assert_allclose(sim1.model.h.corrector_gains, sim2.model.h.corrector_gains)
    np.testing.assert_allclose(sim1.model.v.corrector_gains, sim2.model.v.corrector_gains)

    # Initial observations match
    for o1, o2 in zip(sim1.initial_result.obs_h, sim2.initial_result.obs_h):
        assert o1.ratio == o2.ratio
        assert o1.level == o2.level
    for o1, o2 in zip(sim1.initial_result.obs_v, sim2.initial_result.obs_v):
        assert o1.ratio == o2.ratio
        assert o1.level == o2.level

    # Sequence of commands
    rng_test = np.random.default_rng(1234)
    for _ in range(5):
        cmd_h = rng_test.uniform(-lattice.theta_max, lattice.theta_max, size=lattice.h.n_corr)
        cmd_v = rng_test.uniform(-lattice.theta_max, lattice.theta_max, size=lattice.v.n_corr)

        res1 = sim1.set_correctors(cmd_h, cmd_v)
        res2 = sim2.set_correctors(cmd_h, cmd_v)

        assert res1.s_mean == res2.s_mean
        assert res1.is_lost == res2.is_lost
        assert res1.loss_s == res2.loss_s
        for o1, o2 in zip(res1.obs_h, res2.obs_h):
            assert o1.ratio == o2.ratio
            assert o1.level == o2.level
        for o1, o2 in zip(res1.obs_v, res2.obs_v):
            assert o1.ratio == o2.ratio
            assert o1.level == o2.level

    score1 = compute_reward(s0=sim1.initial_result.s_mean, sf=sim1.last_result.s_mean, shots_used=sim1.shots_used, budget=40)
    score2 = compute_reward(s0=sim2.initial_result.s_mean, sf=sim2.last_result.s_mean, shots_used=sim2.shots_used, budget=40)
    assert score1.reward == score2.reward
