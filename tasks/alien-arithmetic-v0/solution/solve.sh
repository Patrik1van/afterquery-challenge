#!/bin/bash
# Reference run: a scripted experimenter (see solve.py).
set -euo pipefail
python3 "$(dirname "$0")/solve.py"
