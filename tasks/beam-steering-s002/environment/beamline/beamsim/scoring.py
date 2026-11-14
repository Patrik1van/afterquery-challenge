"""Reward calculation and episode scoring according to docs/PLAN.md section 4.

Scoring rules:
- S0 = S_mean at initial setting theta = 0
- Sf = S_mean at final setting (noise-free, true perturbed model)
- progress = clip((Sf - S0) / (0.95 - S0), 0.0, 1.0)
- solved = Sf >= 0.95
- reward = 0.8 * progress + 0.2 * solved * (1.0 - shots_used / budget)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np


@dataclass
class ScoreDetails:
    reward: float
    solved: bool
    s0: float
    sf: float
    progress: float
    shots_used: int
    budget: int
    loss_s: float | None
    is_lost: bool
    s_hat_h: list[float]
    s_hat_v: list[float]

    def to_dict(self) -> dict:
        return asdict(self)


def compute_reward(
    s0: float,
    sf: float,
    shots_used: int,
    budget: int = 40,
    loss_s: float | None = None,
    is_lost: bool = False,
    s_hat_h: np.ndarray | list[float] | None = None,
    s_hat_v: np.ndarray | list[float] | None = None,
) -> ScoreDetails:
    """Compute progress, solved flag, and reward strictly from PLAN §4."""
    s0 = float(s0)
    sf = float(sf)
    shots_used = int(shots_used)
    budget = int(budget)

    # Progress toward solved threshold 0.95
    denom = 0.95 - s0
    if denom > 1e-6:
        progress = float(np.clip((sf - s0) / denom, 0.0, 1.0))
    else:
        progress = 1.0 if sf >= 0.95 else 0.0

    solved = bool(sf >= 0.95)

    # Efficiency bonus: 0.2 * solved * (1 - shots_used / budget)
    # Capped between 0 and 1
    efficiency_factor = max(0.0, 1.0 - float(shots_used) / float(budget))
    efficiency_bonus = 0.2 * float(solved) * efficiency_factor

    reward = float(0.8 * progress + efficiency_bonus)

    sh_h = [float(x) for x in s_hat_h] if s_hat_h is not None else []
    sh_v = [float(x) for x in s_hat_v] if s_hat_v is not None else []

    return ScoreDetails(
        reward=reward,
        solved=solved,
        s0=s0,
        sf=sf,
        progress=progress,
        shots_used=shots_used,
        budget=budget,
        loss_s=loss_s,
        is_lost=is_lost,
        s_hat_h=sh_h,
        s_hat_v=sh_v,
    )
