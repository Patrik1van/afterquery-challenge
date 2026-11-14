#!/bin/bash
# Runs inside the separate verifier container (built from tests/Dockerfile).
set -u
mkdir -p /logs/verifier
python3 /tests/verify.py
