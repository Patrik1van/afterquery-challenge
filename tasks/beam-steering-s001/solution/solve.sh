#!/bin/bash
set -euo pipefail

pip install --no-cache-dir --break-system-packages scipy
python3 /solution/solve.py
