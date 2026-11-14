#!/bin/bash
# Runs in a SEPARATE verifier container, so nothing the agent changed in its own container
# (fake python, edited files) can affect scoring. The score was computed inside the sim
# sidecar by the collect hook and arrives here as an artifact.
mkdir -p /logs/verifier
python3 - <<'EOF'
import json
try:
    r = json.load(open("/sim/out/result.json"))
    score = max(0.0, min(1.0, float(r["score"])))
except Exception as e:
    r, score = {"error": f"no result artifact: {e}"}, 0.0
json.dump(r, open("/logs/verifier/details.json", "w"), indent=1)
open("/logs/verifier/reward.txt", "w").write(f"{score}\n")
print("reward:", score, {k: v for k, v in r.items() if k != "log"})
EOF
