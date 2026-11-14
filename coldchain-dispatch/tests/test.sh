#!/bin/bash
# Runs in a SEPARATE verifier container, so nothing the agent changed in its own container
# (fake python, edited files) can affect scoring. The delivery score was computed inside the sim
# sidecar by the collect hook; both files arrive here as artifacts.
# Hidden-mode shifts: reward = 0.7 * delivery score + 0.3 * share of hidden rules guessed right.
mkdir -p /logs/verifier
python3 - <<'PY'
import json
try:
    r = json.load(open("/sim/out/result.json"))
    score = max(0.0, min(1.0, float(r["score"])))
except Exception as e:
    r, score = {"error": f"no result artifact: {e}"}, 0.0
truth = r.get("truth")
if truth:
    try:
        beliefs = json.load(open("/app/beliefs.json"))
    except Exception:
        beliefs = {}
    if not isinstance(beliefs, dict):
        beliefs = {}
    items = [(m, k) for m in truth for k in ("speed", "cargo")]
    hits = sum(1 for m, k in items
               if isinstance(beliefs.get(m), dict) and beliefs[m].get(k) == truth[m][k])
    acc = hits / len(items)
    r.update(beliefs=beliefs, belief_accuracy=acc, delivery_score=score)
    score = 0.7 * score + 0.3 * acc
json.dump(r, open("/logs/verifier/details.json", "w"), indent=1)
open("/logs/verifier/reward.txt", "w").write(f"{round(score, 4)}\n")
print("reward:", round(score, 4), {k: v for k, v in r.items() if k != "log"})
PY
