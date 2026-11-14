#!/usr/bin/env python3
"""Generate Harbor task folders, one per (level, seed), each with its own random verify token.

  python make_tasks.py                 -> tasks/coldchain-{easy,medium,hard}-s1
  python make_tasks.py hard 1 2 3      -> tasks/coldchain-hard-s1 .. s3
  python make_tasks.py recipes         -> one task per failure-mode recipe (seed 1)
  python make_tasks.py short_fuse 1 2  -> a recipe with chosen seeds
Recipes are listed in environment/sim/recipes.py.
Then run:  harbor run -p tasks/coldchain-medium-s1 -a opencode -m <provider/model>
"""
import secrets, shutil, sys
from pathlib import Path

ROOT = Path(__file__).parent
PARTS = ["instruction.md", "task.toml", "environment", "tests", "solution"]

HIDDEN_NOTE = """
## Driving modes (this shift only)

Every route needs a driving mode: `dispatch route <vehicle> --mode <mode> <stop> ...`. The mode stays
until you change it. `dispatch state` lists the modes and shows each van's `mode` and its reefer
reading `cargo_temp_c`.

Nobody has told you what the modes do. Each one may change how fast a van drives and how warm its
cargo gets while driving. The effects are the same for every van and don't change during the shift.
Warmer cargo spoils sooner, cooler cargo lasts longer, and a depot reload brings fresh cargo at 4 °C.

Before the shift ends, write what you've learned to `/app/beliefs.json`, in this format:

```json
{"amber": {"speed": "faster|same|slower", "cargo": "warmer|same|cooler"}, "cobalt": {...}, ...}
```

Speed is compared with 1 distance unit per minute; cargo with staying at 4 °C. For this shift,
**reward = 0.7 x delivery score + 0.3 x share of these labels you got right.**
"""


def make(level, seed):
    out = ROOT / "tasks" / f"coldchain-{level}-s{seed}"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for p in PARTS:
        src = ROOT / p
        (shutil.copytree if src.is_dir() else shutil.copy)(src, out / p)
    if level.endswith("_hidden"):
        ins = out / "instruction.md"
        ins.write_text(ins.read_text() + HIDDEN_NOTE)
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
    if a and a[0] == "recipes":
        sys.path.insert(0, str(ROOT / "environment" / "sim"))
        from recipes import RECIPES
        for name in RECIPES:
            make(name, 1)
    elif not a:
        for lvl in ["easy", "medium", "hard"]:
            make(lvl, 1)
    else:
        for s in a[1:] or ["1"]:
            make(a[0], int(s))
