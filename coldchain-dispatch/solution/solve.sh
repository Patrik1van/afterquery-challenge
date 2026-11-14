#!/bin/bash
# Reference run: FAIR planner with exactly the agent's information (released orders only; hidden
# modes learned by short probes), plus insertion + local search + restarts every 5 minutes.
# It acts only through the same dispatch CLI as the agent.
# Upper bound instead (knows future orders + true rules):  python3 planner.py dispatch --privileged
python3 "$(dirname "$0")/planner.py" dispatch
