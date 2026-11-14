#!/usr/bin/env python3
"""Check every recipe offline before spending model credits.

For each recipe it plays the reference dispatcher and several "LLM-like" proxy policies, each with
one bad habit we expect frontier agents to show. A good recipe: reference scores high, and the
proxy for the habit that recipe targets scores clearly lower.

  python scripts/probe_recipes.py            # all recipes, seeds 1-3
  python scripts/probe_recipes.py short_fuse 1 2 3 4 5
"""
import math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "environment" / "sim"), str(ROOT / "solution")]
from coldchain import ColdChainEnv, DEPOT, greedy_agent   # noqa: E402
from recipes import RECIPES                                # noqa: E402
import reference                                           # noqa: E402

def caller(env):
    def call(cmd):
        if cmd[0] == "route":
            return env.step("set_route", {"vehicle": cmd[1], "stops": cmd[2:]})
        raise ValueError(cmd)
    return call

def run(env, decide, wait=10):
    while not env.done:
        decide(env, env.step("get_state"))
        env.step("wait", {"minutes": wait})
    return env.reward()["score"]

# ---- policies -------------------------------------------------------------------------------
def p_reference(env):          # replans everything every 10 min
    return run(env, lambda e, s: reference.plan(s, caller(e)))

def p_coarse_wait(env):        # same planner, but checks in only once an hour
    return run(env, lambda e, s: reference.plan(s, caller(e)), wait=60)

def p_nearest_first(env):      # serves whatever is closest; ignores deadlines
    def decide(e, s):
        taken = {x for v in s["vehicles"] for x in v["route"]}
        free = [o for o in s["orders"] if o["status"] == "pending" and o["id"] not in taken]
        for v in s["vehicles"]:
            if v["status"] != "ok" or v["route"] or not free:
                continue
            free.sort(key=lambda o: math.hypot(o["x"] - v["x"], o["y"] - v["y"]))
            o = next((o for o in free if o["demand"] <= v["load"]), None)
            stops = [o["id"]] if o else [DEPOT]
            if o:
                free.remove(o)
            e.step("set_route", {"vehicle": v["id"], "stops": stops})
    return run(env, decide)

def p_plan_once(env):          # makes one good plan, then only appends new orders; ignores events
    state = {"first": True}
    def decide(e, s):
        if state["first"]:
            reference.plan(s, caller(e)); state["first"] = False; return
        taken = {x for v in s["vehicles"] for x in v["route"]}
        seen = state.setdefault("seen", set(o["id"] for o in s["orders"]))
        for o in s["orders"]:
            if o["id"] in seen or o["id"] in taken:
                continue
            seen.add(o["id"])
            v = min(s["vehicles"], key=lambda v: len(v["route"]))   # may be the broken one
            e.step("set_route", {"vehicle": v["id"], "stops": v["route"] + [o["id"]]})
    return run(env, decide)

def p_serve_all(env):          # every order gets a slot, earliest deadline first; nothing is dropped
    def decide(e, s):
        vans = [v for v in s["vehicles"] if v["status"] == "ok"]
        orders = {o["id"]: o for o in s["orders"] if o["status"] == "pending"}
        seqs = {v["id"]: [] for v in vans}
        for n, oid in enumerate(sorted(orders, key=lambda i: orders[i]["tw"][1])):
            seqs[vans[n % len(vans)]["id"]].append(oid)
        for v in vans:
            stops = reference.with_reloads(seqs[v["id"]], orders, v["load"], v["cap"]) or []
            if stops != v["route"]:
                e.step("set_route", {"vehicle": v["id"], "stops": stops})
    return run(env, decide)

def p_no_reload_plan(env):     # plans by urgency but only reloads when the van is empty
    def decide(e, s):
        taken = {x for v in s["vehicles"] for x in v["route"]}
        free = sorted((o for o in s["orders"] if o["status"] == "pending" and o["id"] not in taken),
                      key=lambda o: o["spoil_at"])
        for v in s["vehicles"]:
            if v["status"] != "ok" or v["route"]:
                continue
            stops, load = [], v["load"]
            for o in list(free):
                if o["demand"] <= load:
                    stops.append(o["id"]); load -= o["demand"]; free.remove(o)
            e.step("set_route", {"vehicle": v["id"], "stops": stops or [DEPOT]})
    return run(env, decide)

POLICIES = {"reference": p_reference, "greedy": lambda env: greedy_agent(env)["score"], "nearest_first": p_nearest_first,
            "coarse_wait_60": p_coarse_wait, "plan_once": p_plan_once, "serve_all": p_serve_all,
            "no_reload_plan": p_no_reload_plan}

if __name__ == "__main__":
    names = [sys.argv[1]] if len(sys.argv) > 1 else list(RECIPES)
    seeds = [int(x) for x in sys.argv[2:]] or [1, 2, 3]
    print(f"{'recipe':14s}" + "".join(f"{p:>16s}" for p in POLICIES))
    for n in names:
        row = []
        for p, f in POLICIES.items():
            sc = [f(ColdChainEnv(n, s, max_tool_calls=10_000)) for s in seeds]
            row.append(sum(sc) / len(sc))
        print(f"{n:14s}" + "".join(f"{x:16.3f}" for x in row))
