#!/usr/bin/env bash
# Run full judge evaluation over tasks/beam-steering-s000..s004 on Modal
# Runs oracle, nop, stop-early, and 3x opencode (Gemini 3.8-Flash), then prints summary table.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# 1. Set PYTHONPATH
export PYTHONPATH="$REPO_ROOT"

# 2. Export API keys from .env if present (never print)
if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    source "$REPO_ROOT/.env"
    set +a
fi

echo "=================================================================="
echo "CERN TT20 Beam Steering Benchmark — Harbor Modal Evaluation"
echo "Tasks: tasks/beam-steering-s000..s004"
echo "Concurrency: 5"
echo "=================================================================="

# 3. Oracle (Bayes reference)
echo -e "\n>>> [1/4] Running Oracle Agent (Bayes reference) on tasks s000..s004..."
harbor run -p tasks -i "beam-steering-s*" -n 5 -e modal -a oracle --env-file .env

# 4. Nop (Do-nothing baseline)
echo -e "\n>>> [2/4] Running Nop Agent (Do-nothing baseline) on tasks s000..s004..."
harbor run -p tasks -i "beam-steering-s*" -n 5 -e modal -a nop --env-file .env

# 5. Stop-Early baseline
echo -e "\n>>> [3/4] Running Stop-Early Agent baseline on tasks s000..s004..."
harbor run -p tasks -i "beam-steering-s*" -n 5 -e modal -a agents.baselines:StopEarlyAgent --env-file .env

# 6. OpenCode (Gemini 3.8-Flash with -k 3)
echo -e "\n>>> [4/4] Running OpenCode + Gemini 3.8-Flash (k=3) on tasks s000..s004..."
harbor run -p tasks -i "beam-steering-s*" -n 5 -e modal -a opencode -m gemini/gemini-3.8-flash --api-key-name GEMINI_API_KEY -k 3 --env-file .env

# 7. Export trajectories for webapp replay
echo -e "\n>>> Exporting trajectories..."
python3 scripts/export_trajectories.py

# 8. Print results summary table
echo -e "\n>>> Generating Results Table..."
python3 scripts/summarize_results.py
