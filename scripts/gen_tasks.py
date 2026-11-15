"""Task generator for Harbor beamline tasks (tasks/beam-steering-sNNN).

Generates tasks with:
- Multi-container environment (agent main container + beamline FastMCP sidecar)
- Token-protected GET /score endpoint on sidecar
- Agent container with python3 + numpy + curl only (no beamsim, no token)
- Tests verifier curling /score and writing reward.txt + details.json
- Solution running Bayes agent as an MCP client
All files written strictly with LF line endings (newline='\\n').
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import secrets
import shutil

REPO_ROOT = Path(__file__).resolve().parent.parent

INSTRUCTION_TEMPLATE = """# Operator Briefing: CERN TT20 Beam Steering Task

You are an expert accelerator operator controlling a simulated proton transfer line (~900 meters long) modeled after the CERN TT20 beamline.

## 1. Goal
Your mission is to steer, thread, and centre the proton beam across the entire transfer line through all Secondary Emission Monitors (SEMs) in both the horizontal and vertical planes within a limited budget of **{budget} shots**.

An episode is **solved** when the true mean intensity-normalized symmetry $S_{\\text{{mean}}} \\ge 0.95$.

### Reward Formula
The evaluation score is computed server-side from true, noise-free state:
- $S_0$: noise-free mean symmetry at initial state $\\theta = 0$ (typically near 0 due to missteered beam).
- $S_f$: noise-free mean symmetry at your final submitted setting.
- $\\text{{progress}} = \\operatorname{{clip}}\\left(\\frac{{S_f - S_0}}{{0.95 - S_0}}, 0.0, 1.0\\right)$
- $\\text{{solved}} = (S_f \\ge 0.95)$
- $\\text{{reward}} = 0.8 \\cdot \\text{{progress}} + 0.2 \\cdot \\text{{solved}} \\cdot \\left(1 - \\frac{{\\text{{shots_used}}}}{{\\text{{budget}}}}\\right)$

Notice that doing nothing gives 0 reward. Making real progress towards threading and centring gives up to 0.8, and solving with fewer shots earns an efficiency bonus up to 0.2.

---

## 2. Physics in Brief
1. **Response Matrix & Trajectory**:
   Beam deflection is governed by linear optics per plane:
   $$x_h = R_h (\\Delta\\theta_h - \\theta_h), \\qquad x_v = R_v (\\Delta\\theta_v - \\theta_v)$$
   where $\\Delta\\theta$ are unknown hidden missteering kicks, and $\\theta$ are your commanded corrector dipole angles.
   - Horizontal: 10 correctors, 11 SEM monitors.
   - Vertical: 12 correctors, 10 SEM monitors.
   - The last 1–2 correctors in each plane only influence the final monitor.

2. **Secondary Emission Monitors (SEMs)**:
   Each SEM monitor has two split foils spanning $[-L, 0]$ and $[0, L]$ with $L \\approx 10\\sigma$.
   - **Ratio** $r = \\tilde{{I}}_1 / (\\tilde{{I}}_1 + \\tilde{{I}}_2)$: Measures relative beam position across the two foils. For a centred beam ($x = 0$), $r \\approx 0.5$. If beam is deflected positive ($x > 0$), $r < 0.5$; if $x < 0$, $r > 0.5$.
   - **Level** $\\ell = (\\tilde{{I}}_1 + \\tilde{{I}}_2) / I_{{\\max}}$: Measures total intercepted intensity relative to a centred beam. When beam passes through the monitor, $\\ell \\approx 1.0$. When beam misses foils or is lost upstream, the foils read only background noise ($\\ell \\approx \\nu \\approx 0.25$).

3. **Joint Beam Loss**:
   The beampipe has aperture half-widths $a_h, a_v \\sim 25\\text{{–}}35\\text{{ mm}}$. If $|x_h| > a_h$ OR $|x_v| > a_v$ at any longitudinal point along the line, the beam hits the wall and is lost. **All monitors in both planes downstream of the loss point detect background noise only.**

4. **Nominal Model vs Real Machine**:
   The response matrices and beam sizes provided by `get_machine_info()` are **nominal**. The true machine operates with realistic domain mismatch (slight corrector gain errors ~±2%, optical beta-beating ~30–50%, background noise variations). Do not rely solely on open-loop inversion—use observed monitor readings to iteratively refine your steering.

---

## 3. Available MCP Tools
The beamline environment is connected as an MCP server named `beamline` exposing 4 tools:

1. **`get_machine_info()`** (Costs 0 shots):
   Returns the nominal model: element coordinates, nominal Twiss parameters, nominal response matrices $R_{{\\text{{sem}}}}$ and $R_{{\\text{{ap}}}}$ for both planes, apertures, noise level $\\nu=0.25$, setting bound $\\theta_{{\\max}}$, budget ({budget}), and remaining shots.
2. **`set_correctors(h: list[float], v: list[float])`** (Costs 1 shot):
   Applies absolute corrector dipole deflection angles (10 floats for horizontal, 12 floats for vertical) clipped to $[-\\theta_{{\\max}}, \\theta_{{\\max}}]$. Returns noisy monitor readings $\\{r, \\ell\\}$ for each SEM in H and V, plus `shots_remaining`.
3. **`get_history()`** (Costs 0 shots):
   Returns the complete chronological log of all your previous settings and the corresponding monitor readings.
4. **`submit()`** (Costs 0 shots):
   Freezes the episode when you are satisfied that the beam is centred. No further corrector commands are accepted. Your last commanded setting is evaluated for the final score.

You have access to a full Linux shell with Python and NumPy installed. You can write and run scripts to analyze data, compute matrix inversions, run optimization, or interact with the tools.

---

## 4. Operational Strategy
- Start by calling `get_machine_info()` and examining the initial state.
- Write Python scripts to compute candidate corrections based on the response matrices and monitor readings.
- Progressively thread the beam from upstream monitors downstream: correct early monitors first so the beam reaches downstream monitors.
- Once the beam reaches all monitors, fine-tune the correctors to bring ratios $r \\to 0.5$ and levels $\\ell \\to 1.0$.
- When confident, call `submit()`.
"""

TASK_TOML_TEMPLATE = """schema_version = "1.1"

[task]
name = "afterquery/beam-steering-s{seed:03d}"
description = "Steer and centre simulated CERN TT20 beamline via MCP tools"
authors = []
keywords = []

[metadata]
seed = {seed}
budget = {budget}
randomization = "{randomization}"

[verifier]
timeout_sec = 600.0

[agent]
timeout_sec = 3600.0

[environment]
build_timeout_sec = 600.0
cpus = 2
memory_mb = 4096
storage_mb = 10240
gpus = 0

[[environment.mcp_servers]]
name = "beamline"
transport = "streamable-http"
url = "http://beamline:8000/mcp"
args = []

[verifier.env]

[environment.env]

[solution.env]
"""

AGENT_DOCKERFILE = """FROM ubuntu:24.04

RUN apt-get update && apt-get install -y \\
    curl \\
    python3 \\
    python3-pip \\
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --break-system-packages numpy

WORKDIR /app
"""

DOCKER_COMPOSE_TEMPLATE = """# Multi-container Compose configuration for Harbor beamline task
services:
  main:
    depends_on:
      beamline:
        condition: service_healthy

  beamline:
    build:
      context: ./beamline
    expose:
      - "8000"
    healthcheck:
      test: ["CMD", "python", "-c", "import socket; s=socket.create_connection(('localhost',8000),timeout=2); s.close()"]
      interval: 2s
      timeout: 5s
      retries: 15
      start_period: 5s
"""

BEAMLINE_DOCKERFILE_TEMPLATE = """FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy scipy fastmcp

COPY beamsim/ /app/beamsim/
COPY server.py /app/server.py

ENV BEAMLINE_SEED={seed}
ENV BEAMLINE_SALT={salt}
ENV BEAMLINE_BUDGET={budget}
ENV BEAMLINE_RANDOMIZATION={randomization}
ENV BEAMLINE_SCORE_TOKEN={score_token}

EXPOSE 8000

CMD ["python", "server.py"]
"""

TEST_SH_TEMPLATE = """#!/bin/bash
set -euo pipefail

mkdir -p /logs/verifier
curl -s "http://beamline:8000/score?token={score_token}" > /logs/verifier/details.json

python3 -c "
import json
with open('/logs/verifier/details.json', 'r') as f:
    data = json.load(f)
reward = data.get('reward', 0.0)
with open('/logs/verifier/reward.txt', 'w') as f:
    f.write(f'{float(reward):.6f}\\n')
"
"""

SOLVE_SH = """#!/bin/bash
set -euo pipefail

pip install --no-cache-dir --break-system-packages scipy
python3 /solution/solve.py
"""

SOLVE_PY_TEMPLATE = """import json
import urllib.request
from bayes import BayesAgent


class SyncMCPBeamlineClient:
    def __init__(self, url="http://beamline:8000/mcp"):
        self.url = url
        self.headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        self.id = 1
        self.session_id = None
        self._init()

    def _init(self):
        body = json.dumps({
            "jsonrpc": "2.0",
            "id": self.id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "bayes-oracle", "version": "1.0"},
            },
        }).encode("utf-8")
        req = urllib.request.Request(self.url, data=body, headers=self.headers)
        with urllib.request.urlopen(req) as resp:
            self.session_id = resp.headers.get("mcp-session-id")

    def _send(self, method: str, params: dict) -> dict:
        self.id += 1
        body = json.dumps({"jsonrpc": "2.0", "id": self.id, "method": method, "params": params}).encode("utf-8")
        hdrs = dict(self.headers)
        if self.session_id:
            hdrs["mcp-session-id"] = self.session_id
        req = urllib.request.Request(self.url, data=body, headers=hdrs)
        with urllib.request.urlopen(req) as resp:
            text = resp.read().decode("utf-8")
            for line in text.splitlines():
                if line.startswith("data: "):
                    return json.loads(line[6:]).get("result", {})
        return {}

    def get_machine_info(self) -> dict:
        res = self._send("tools/call", {"name": "get_machine_info", "arguments": {}})
        return json.loads(res["content"][0]["text"])

    def set_correctors(self, h, v) -> dict:
        h_list = [float(x) for x in h]
        v_list = [float(x) for x in v]
        res = self._send("tools/call", {"name": "set_correctors", "arguments": {"h": h_list, "v": v_list}})
        return json.loads(res["content"][0]["text"])

    def get_history(self) -> list:
        res = self._send("tools/call", {"name": "get_history", "arguments": {}})
        return json.loads(res["content"][0]["text"])

    def submit(self) -> dict:
        res = self._send("tools/call", {"name": "submit", "arguments": {}})
        return json.loads(res["content"][0]["text"])


def main():
    client = SyncMCPBeamlineClient()
    agent = BayesAgent(n_particles=100_000, chunk_size=25_000, roughening=0.2, seed={seed} + 1_000_003)
    agent.solve(client)
    print("Bayes Oracle finished episode.")


if __name__ == "__main__":
    main()
"""


def write_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def generate_task(seed: int, budget: int = 60, randomization: str = "medium") -> Path:
    task_dir = REPO_ROOT / "tasks" / f"beam-steering-s{seed:03d}"
    score_token = secrets.token_hex(16)
    salt = secrets.token_hex(16)
    existing_test = task_dir / "tests" / "test.sh"
    if existing_test.exists():
        import re
        m = re.search(r"token=([0-9a-fA-F]+)", existing_test.read_text())
        if m:
            score_token = m.group(1)

    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)

    # 1. task.toml
    write_file(
        task_dir / "task.toml",
        TASK_TOML_TEMPLATE.format(seed=seed, budget=budget, randomization=randomization),
    )

    # 2. instruction.md
    write_file(
        task_dir / "instruction.md",
        INSTRUCTION_TEMPLATE.replace("{budget}", str(budget)),
    )

    # 3. environment/Dockerfile (agent container main)
    write_file(task_dir / "environment" / "Dockerfile", AGENT_DOCKERFILE)

    # 4. environment/docker-compose.yaml
    write_file(task_dir / "environment" / "docker-compose.yaml", DOCKER_COMPOSE_TEMPLATE)

    # 5. environment/beamline/
    beamline_dir = task_dir / "environment" / "beamline"
    write_file(
        beamline_dir / "Dockerfile",
        BEAMLINE_DOCKERFILE_TEMPLATE.format(
            seed=seed, salt=salt, budget=budget, randomization=randomization, score_token=score_token
        ),
    )
    # copy beamsim
    shutil.copytree(
        REPO_ROOT / "beamsim",
        beamline_dir / "beamsim",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    # copy server.py
    shutil.copy(REPO_ROOT / "mcp_server" / "server.py", beamline_dir / "server.py")

    # 6. tests/test.sh
    test_sh_path = task_dir / "tests" / "test.sh"
    write_file(test_sh_path, TEST_SH_TEMPLATE.replace("{score_token}", score_token))
    os.chmod(test_sh_path, 0o755)

    # 7. solution/
    solution_dir = task_dir / "solution"
    solve_sh_path = solution_dir / "solve.sh"
    write_file(solve_sh_path, SOLVE_SH)
    os.chmod(solve_sh_path, 0o755)
    write_file(solution_dir / "solve.py", SOLVE_PY_TEMPLATE.replace("{seed}", str(seed)))

    # Copy bayes.py to solution/bayes.py with env_api dependency removed
    bayes_src = (REPO_ROOT / "agents" / "bayes.py").read_text(encoding="utf-8")
    bayes_src = bayes_src.replace(
        "from beamsim.env_api import BeamlineEnv",
        "# beamsim.env_api replaced for standalone MCP execution",
    ).replace(
        "env: BeamlineEnv",
        "env: Any",
    )
    write_file(solution_dir / "bayes.py", bayes_src)

    return task_dir


def main():
    parser = argparse.ArgumentParser(description="Generate Harbor beam-steering tasks.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="Seeds to generate")
    parser.add_argument("--budget", type=int, default=60, help="Shot budget per episode")
    parser.add_argument("--randomization", type=str, default="medium", help="Randomization spread")
    args = parser.parse_args()

    for seed in args.seeds:
        td = generate_task(seed=seed, budget=args.budget, randomization=args.randomization)
        print(f"Generated task: {td.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

