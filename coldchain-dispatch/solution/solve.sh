#!/bin/bash
# Reference run: privileged "clairvoyant" oracle. It rebuilds the exact shift from params.json
# (written by make_tasks.py), so it knows future orders and the true hidden rules, and plans with
# insertion + local search. It still acts only through the same dispatch CLI as the agent.
# Fallback: the non-privileged reference (learns hidden modes by experimenting).
DIR="$(dirname "$0")"
if [ -f "$DIR/params.json" ]; then
  python3 "$DIR/oracle.py" dispatch
else
  python3 "$DIR/reference.py" dispatch
fi
