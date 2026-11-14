#!/bin/bash
set -euo pipefail

mkdir -p /logs/verifier
curl -s "http://beamline:8000/score?token=54edbe6c59149752f315a602c176a008" > /logs/verifier/details.json

python3 -c "
import json
with open('/logs/verifier/details.json', 'r') as f:
    data = json.load(f)
reward = data.get('reward', 0.0)
with open('/logs/verifier/reward.txt', 'w') as f:
    f.write(f'{float(reward):.6f}\n')
"
