#!/usr/bin/env python3
"""Generate Harbor task folders, one per (level, seed), each with its own random verify token.

  python make_tasks.py                 -> tasks/coldchain-{easy,medium,hard}-s1
  python make_tasks.py hard 1 2 3      -> tasks/coldchain-hard-s1 .. s3
Then run:  harbor run -p tasks/coldchain-medium-s1 -a opencode -m <provider/model>
"""
import secrets, shutil, sys
from pathlib import Path

ROOT = Path(__file__).parent
PARTS = ["instruction.md", "task.toml", "environment", "tests", "solution"]

def make(level, seed):
    out = ROOT / "tasks" / f"coldchain-{level}-s{seed}"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for p in PARTS:
        src = ROOT / p
        (shutil.copytree if src.is_dir() else shutil.copy)(src, out / p)
    token = secrets.token_hex(16)
    compose = out / "environment" / "docker-compose.yaml"
    compose.write_text(compose.read_text()
                       .replace('LEVEL: "medium"', f'LEVEL: "{level}"')
                       .replace('SEED: "1"', f'SEED: "{seed}"')
                       .replace("REPLACE_TOKEN", token))
    for f in [out / "tests" / "test.sh", out / "solution" / "solve.sh", out / "environment" / "dispatch"]:
        f.chmod(0o755)
    print("wrote", out)

if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        for lvl in ["easy", "medium", "hard"]:
            make(lvl, 1)
    else:
        for s in a[1:] or ["1"]:
            make(a[0], int(s))
