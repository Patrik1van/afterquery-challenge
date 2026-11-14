#!/bin/bash
# Reference run: rolling-horizon cheapest insertion, using the same dispatch CLI as the agent.
python3 "$(dirname "$0")/reference.py" dispatch
