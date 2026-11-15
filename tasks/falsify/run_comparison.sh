#!/bin/bash
# Compare the reference solver with Gemini 4 Hydrogen and Gemini 4 Carbon on one puzzle set.
#
# Usage (from anywhere in the repo):
#   GEMINI_API_KEY=... tasks/falsify/run_comparison.sh            # reference + Hydrogen + Carbon
#   GEMINI_API_KEY=... tasks/falsify/run_comparison.sh --quick    # reference + Hydrogen only
#   Options: --seed N (default 42), --trials K (default 5)
set -euo pipefail

SEED=42
TRIALS=5
MODELS=(gemini-4-hydrogen gemini-4-carbon)
while [ $# -gt 0 ]; do
  case "$1" in
    --quick) MODELS=(gemini-4-hydrogen); shift ;;
    --seed) SEED="$2"; shift 2 ;;
    --trials) TRIALS="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
: "${GEMINI_API_KEY:?set GEMINI_API_KEY to a Gemini API key}"

TASK_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$TASK_DIR/../.."
STAMP=$(date +%Y%m%d-%H%M%S)
export FALSIFY_SEED="$SEED"
JOBS=()

run() {
  local name="falsify-seed$SEED-$1-$STAMP"
  shift
  echo "== $name"
  uv run harbor run -p "$TASK_DIR" -e docker --job-name "$name" "$@"
  JOBS+=("jobs/$name")
}

run reference -a oracle
for m in "${MODELS[@]}"; do
  run "${m#gemini-4-}" -a opencode -m "google/$m" -k "$TRIALS" -n "$TRIALS" \
    --ae GEMINI_API_KEY="$GEMINI_API_KEY" --ae GOOGLE_GENERATIVE_AI_API_KEY="$GEMINI_API_KEY"
done

python3 - "${JOBS[@]}" <<'EOF'
import json, sys
from pathlib import Path

print(f"\n{'run':48} {'mean':>5}  {'perfect':>7}  rewards")
for job in sys.argv[1:]:
    rewards = []
    for trial in sorted(Path(job).glob("*/")):
        f = trial / "verifier" / "reward.txt"
        rewards.append(float(f.read_text()) if f.exists() else 0.0)
    if rewards:
        mean = sum(rewards) / len(rewards)
        perfect = sum(r == 1.0 for r in rewards)
        print(f"{Path(job).name:48} {mean:5.2f}  {perfect:>3}/{len(rewards):<3}  {' '.join(f'{r:.2f}' for r in rewards)}")
print("\nPer-trial diagnostics: jobs/<run>/<trial>/verifier/details.json")
EOF
