"""FastMCP server exposing the CERN TT20 beamline environment as MCP tools.

Global state per server process. Seed, budget, and randomization configured via env vars.
Exposes tools:
- get_machine_info() -> dict
- set_correctors(h: list[float], v: list[float]) -> dict
- get_history() -> list[dict]
- submit() -> dict

Exposes custom HTTP route:
- GET /score?token=... -> JSON details and freezes episode
"""

from __future__ import annotations

import os
from typing import Any

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from beamsim.env_api import BeamlineEnv, evaluate_env, get_env_true_trajectory

seed = int(os.environ.get("BEAMLINE_SEED", "0"))
budget = int(os.environ.get("BEAMLINE_BUDGET", "60"))
randomization = os.environ.get("BEAMLINE_RANDOMIZATION", "medium")
score_token = os.environ.get("BEAMLINE_SCORE_TOKEN", "")

env = BeamlineEnv(
    seed=seed,
    budget=budget,
    randomization=randomization,
)

mcp = FastMCP("beamline")


@mcp.tool()
def get_machine_info() -> dict[str, Any]:
    """Return nominal lattice and machine parameters (H/V correctors, SEMs, apertures, noise, budget)."""
    return env.get_machine_info()


@mcp.tool()
def set_correctors(h: list[float], v: list[float]) -> dict[str, Any]:
    """Apply corrector kicks in radians (10 horizontal, 12 vertical). Costs 1 shot. Returns SEM readings."""
    return env.set_correctors(h=h, v=v)


@mcp.tool()
def get_history() -> list[dict[str, Any]]:
    """Return full history of commanded settings and observed SEM monitor readings."""
    return env.get_history()


@mcp.tool()
def submit() -> dict[str, Any]:
    """Freeze the episode. The last commanded corrector settings are evaluated for final score."""
    return env.submit()


@mcp.custom_route("/score", methods=["GET"])
async def score_endpoint(request: Request) -> Response:
    token = request.query_params.get("token")
    if not score_token or token != score_token:
        return JSONResponse({"error": "Unauthorized: invalid or missing score token"}, status_code=403)
    env.submit()
    score_details = evaluate_env(env)
    traj = get_env_true_trajectory(env)
    res = score_details.to_dict()
    res["true_trajectory"] = traj
    return JSONResponse(res)


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
