#!/bin/bash
set -euo pipefail

mkdir -p /logs/verifier
curl -s "http://beamline:8000/score?token=4de1d73b77b66da9a6615cc0e17c9ed3" > /logs/verifier/details.json

python3 -c "
import json
with open('/logs/verifier/details.json', 'r') as f:
    data = json.load(f)
reward = data.get('reward', 0.0)
with open('/logs/verifier/reward.txt', 'w') as f:
    f.write(f'{float(reward):.6f}\n')
"
