#!/usr/bin/env python3
"""Check every recipe offline before spending model credits.

For each recipe it plays the reference dispatcher and several "LLM-like" proxy policies, each with
one bad habit we expect frontier agents to show. On *_hidden recipes every proxy except the
reference and the oracle keeps the first mode it picks and never experiments; their belief score
is 0 (no beliefs written). A good recipe: reference scores high, and the
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
import reference, planner                                  # noqa: E402

def caller(env):
    def call(cmd):
        if cmd[0] == "route":
            body, stops = {"vehicle": cmd[1]}, cmd[2:]
            if stops[:1] == ["--mode"]:
                body["mode"], stops = stops[1], stops[2:]
            body["stops"] = stops
            return env.step("set_route", body)
        raise ValueError(cmd)
    return call

def belief_accuracy(beliefs, truth):
    if not truth:
        return None
    items = [(m, k) for m in truth for k in ("speed", "cargo")]
    ok = sum(1 for m, k in items if (beliefs or {}).get(m, {}).get(k) == truth[m][k])
    return ok / len(items)

def final(env, beliefs=None):
    """Same formula as tests/test.sh: hidden-rule shifts weight beliefs at 30%."""
    sc, acc = env.reward()["score"], belief_accuracy(beliefs, env.truth())
    return sc if acc is None else 0.7 * sc + 0.3 * acc

def fix_first_mode(env):
    """Policies that never experiment just keep whichever mode they pick first."""
    if env.modes:
        for v in env.vehicles:
            env.step("set_route", {"vehicle": v, "stops": [], "mode": sorted(env.modes)[0]})

def run(env, decide, wait=10):
    fix_first_mode(env)
    while not env.done:
        decide(env, env.step("get_state"))
        env.step("wait", {"minutes": wait})
    return final(env)

# ---- policies -------------------------------------------------------------------------------
def _planner(env, privileged):
    b = planner.run(lambda: env.step("get_state"), lambda m: env.step("wait", {"minutes": m}),
                    caller(env), lambda: env.step("finish"), env.level, env.seed,
                    privileged=privileged, noisy=bool(env.noise))
    return final(env, b)

def p_reference(env):          # FAIR reference (solve.sh): agent's information only, strong optimiser
    return _planner(env, privileged=False)

def p_oracle(env):             # upper bound: knows future orders and true hidden rules
    return _planner(env, privileged=True)

def p_learn_once(env):         # fair reference that never re-checks what it learned
    b = planner.run(lambda: env.step("get_state"), lambda m: env.step("wait", {"minutes": m}),
                    caller(env), lambda: env.step("finish"), env.level, env.seed, recheck=False,
                    noisy=bool(env.noise))
    return final(env, b)

def p_simple_ref(env):         # first reference: one insertion pass, slow mode rotation
    b = reference.run(lambda: env.step("get_state"), lambda m: env.step("wait", {"minutes": m}),
                      caller(env), lambda: env.step("finish"))
    return final(env, b)

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

POLICIES = {"oracle": p_oracle, "reference": p_reference, "learn_once": p_learn_once,
            "simple_ref": p_simple_ref, "greedy": lambda env: (fix_first_mode(env), greedy_agent(env), final(env))[-1], "nearest_first": p_nearest_first,
            "coarse_wait_60": p_coarse_wait, "plan_once": p_plan_once, "serve_all": p_serve_all,
            "no_reload_plan": p_no_reload_plan}

def _cell(args):
    n, p, s = args
    return n, p, POLICIES[p](ColdChainEnv(n, s, max_tool_calls=10_000_000))

if __name__ == "__main__":
    from multiprocessing import Pool
    names = [sys.argv[1]] if len(sys.argv) > 1 else list(RECIPES)
    seeds = [int(x) for x in sys.argv[2:]] or [1, 2, 3]
    jobs = [(n, p, s) for n in names for p in POLICIES for s in seeds]
    with Pool() as pool:
        res = pool.map(_cell, jobs)
    agg = {}
    for n, p, sc in res:
        agg.setdefault((n, p), []).append(sc)
    print(f"{'recipe':20s}" + "".join(f"{p:>15s}" for p in POLICIES))
    for n in names:
        print(f"{n:20s}" + "".join(f"{sum(agg[n, p]) / len(agg[n, p]):15.3f}" for p in POLICIES))
