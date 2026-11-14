#!/usr/bin/env python3
"""Privileged reference ("clairvoyant oracle") for the reproducible reference run.

Unlike an agent, it rebuilds the exact shift from the task's level + seed, so it knows
  - every order in advance, including ones not released yet, and
  - the true effect of every hidden driving mode.
Every 5 simulated minutes it re-plans all routes with insertion + local search over the
whole remaining shift (released and future orders), then sends each van the released
prefix of its plan through the same `dispatch` CLI the agent uses. It can still only act
through legal commands; what it has extra is information. The gap between an agent and
this oracle is the value of foresight and of knowing the hidden rules.

Usage: python3 oracle.py [dispatch-binary]   (needs params.json + simulator sources next to it)
"""
import json, math, os, random, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELOAD, SERVICE, STEP = 10.0, 5.0, 5
DIST_W = 2.0          # 1 on-time order ~ 500 distance units in the final score
BELIEFS = os.environ.get("BELIEFS_PATH", "/app/beliefs.json")

def load_instance(level, seed):
    sys.path.insert(0, str(HERE / "sim"))
    sys.path.insert(0, str(HERE.parent / "environment" / "sim"))   # when run from the repo
    from coldchain import ColdChainEnv
    env = ColdChainEnv(level, seed)
    orders = {o.id: dict(id=o.id, x=o.x, y=o.y, demand=o.demand, release=o.release,
                         tw=[o.tw_open, o.tw_close], spoil_at=o.spoil_at) for o in env.orders.values()}
    return orders, env.modes, env.truth()

# ---------------------------------------------------------------- route model
def simulate(v, t, seq, orders, mode):
    """Insert DEPOT reloads as needed, simulate; return (value, dist, stops) or None if infeasible."""
    sp, rate = mode
    x, y, now = v["x"], v["y"], max(t, v["busy_until"])
    load, cap = v["load"], v["cap"]
    shelf = -10.0 * (v.get("cargo_temp_c", 4.0) - 4.0)
    value = dist = 0.0
    stops = []
    for oid in seq:
        o = orders[oid]
        if o["demand"] > cap:
            return None
        if o["demand"] > load:                       # reload first
            leg = math.hypot(x, y); dt = leg / sp
            x, y, dist, now = 0.0, 0.0, dist + leg, now + dt + RELOAD
            load, shelf = cap, 0.0
            stops.append("DEPOT")
        leg = math.hypot(o["x"] - x, o["y"] - y); dt = leg / sp
        x, y, dist, now = o["x"], o["y"], dist + leg, now + dt
        shelf += rate * dt
        start = max(now, o["tw"][0], o["release"])
        if start <= o["spoil_at"] + shelf:
            value += 1.0 if start <= o["tw"][1] else 0.5
        now = start + SERVICE
        load -= o["demand"]
        stops.append(oid)
    return value, dist, stops

def route_cost(v, t, seq, orders, modes):
    """Best (objective, value, dist, stops, mode_label) over allowed modes."""
    best = None
    for label, m in modes.items():
        r = simulate(v, t, seq, orders, m)
        if r is None:
            continue
        val, dist, stops = r
        obj = val * 1000 - dist * DIST_W
        if best is None or obj > best[0]:
            best = (obj, val, dist, stops, label)
    return best

# ---------------------------------------------------------------- optimiser
def optimise(vehicles, t, pool, orders, modes, rng, iters=3, warm=None):
    seqs = {v["id"]: [] for v in vehicles}
    if warm:                                         # keep the previous plan as a starting point
        live = set(pool)
        for vid in seqs:
            seqs[vid] = [o for o in warm.get(vid, []) if o in live]
    vmap = {v["id"]: v for v in vehicles}
    cost = {vid: route_cost(vmap[vid], t, seqs[vid], orders, modes) for vid in seqs}
    for vid in seqs:                                 # warm plan may have become infeasible
        if cost[vid] is None:
            seqs[vid] = []
            cost[vid] = route_cost(vmap[vid], t, [], orders, modes)
    placed = {o for s in seqs.values() for o in s}

    def try_insert(oid):
        best = None
        for vid, s in seqs.items():
            base = cost[vid][0]
            for pos in range(len(s) + 1):
                cand = s[:pos] + [oid] + s[pos:]
                c = route_cost(vmap[vid], t, cand, orders, modes)
                if c and c[0] - base > 0 and (best is None or c[0] - base > best[0]):
                    best = (c[0] - base, vid, cand, c)
        if best:
            _, vid, cand, c = best
            seqs[vid], cost[vid] = cand, c
            return True
        return False

    noise = 0.0 if rng.random() < 0.25 else 40.0      # restart 0 is deterministic-ish; others explore
    order_ids = sorted((i for i in pool if i not in placed),
                       key=lambda i: orders[i]["spoil_at"] + rng.gauss(0, noise))
    unassigned = [o for o in order_ids if not try_insert(o)]
    total = lambda: sum(c[0] for c in cost.values())
    for _ in range(iters):
        before = total()
        # relocate: pull each order out and put it back at its best spot (or drop it if it
        # no longer pays for itself). Never lowers the objective.
        for vid in list(seqs):
            for oid in list(seqs[vid]):
                seqs[vid] = [x for x in seqs[vid] if x != oid]
                cost[vid] = route_cost(vmap[vid], t, seqs[vid], orders, modes)
                if not try_insert(oid):
                    unassigned.append(oid)
        unassigned = [oid for oid in unassigned if not try_insert(oid)]
        if total() <= before + 1e-6:
            break
    total = sum(c[0] for c in cost.values())
    return total, seqs, cost

MEMORY = {}

def plan(state, orders, modes, call, rng, restarts=16):
    t = state["t"]
    vehicles = [v for v in state["vehicles"] if v.get("status", "ok") == "ok"]
    status = {o["id"]: o["status"] for o in state["orders"]}
    pool = [oid for oid, o in orders.items()
            if status.get(oid, "pending") == "pending" and o["spoil_at"] > t]
    modes = modes or {None: {"speed": 1.0, "shelf_rate": 0.0}}
    mtab = {k: (m["speed"], m["shelf_rate"]) for k, m in modes.items()}
    best = None
    for r in range(restarts):
        warm = MEMORY.get("seqs") if r == 0 else None
        total, seqs, cost = optimise(vehicles, t, pool, orders, mtab, random.Random(rng.random() + r),
                                     warm=warm)
        if best is None or total > best[0]:
            best = (total, seqs, cost)
    _, seqs, cost = best
    MEMORY["seqs"] = seqs
    for v in vehicles:
        c = cost[v["id"]]
        stops, label = c[3], c[4]
        sent = []
        for s in stops:                              # only released orders can be routed now
            if s != "DEPOT" and orders[s]["release"] > t:
                break
            sent.append(s)
        args = ["route", v["id"]] + (["--mode", label] if label is not None else []) + sent
        if sent != v["route"] or (label is not None and label != v.get("mode")):
            call(args)

def run(state_fn, wait_fn, call, finish_fn, level, seed, write_beliefs=None):
    MEMORY.clear()
    orders, modes, truth = load_instance(level, seed)
    rng = random.Random(seed)
    s = state_fn()
    while not s.get("error") and s["t"] < s["horizon"]:
        plan(s, orders, modes, call, rng)
        r = wait_fn(STEP)
        if r.get("done") or r.get("error"):
            break
        s = state_fn()
    if truth and write_beliefs:
        write_beliefs(truth)
    finish_fn()
    return truth

def main():
    dispatch = sys.argv[1] if len(sys.argv) > 1 else "dispatch"
    d = lambda cmd: json.loads(subprocess.run([dispatch, *cmd], capture_output=True, text=True).stdout)
    p = json.loads((HERE / "params.json").read_text())

    def write(b):
        try:
            Path(BELIEFS).write_text(json.dumps(b, indent=1))
        except OSError:
            pass
    run(lambda: d(["state"]), lambda m: d(["wait", str(m)]), d, lambda: d(["finish"]),
        p["level"], int(p["seed"]), write)

if __name__ == "__main__":
    main()
