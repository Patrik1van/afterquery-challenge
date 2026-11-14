#!/usr/bin/env python3
"""Reference dispatcher: rolling-horizon cheapest insertion, plus experiment-then-exploit for
shifts with hidden driving modes.

Every 10 simulated minutes it rebuilds all routes from scratch:
  - orders sorted by spoilage deadline (most urgent first)
  - each order inserted at the vehicle/position that keeps the most deliveries on time
    and adds the least driving; DEPOT reloads are inserted automatically when needed.
If the shift has driving modes, it first rotates the modes across vans and measures clean
10-minute driving intervals (distance moved, reefer temperature change) to estimate each mode's
speed and effect on cargo, writes those beliefs to beliefs.json, then drives in the best mode.
It talks to the simulator only through the same `dispatch` CLI the agent uses.
"""
import json, math, os, subprocess, sys

DISPATCH = sys.argv[1] if len(sys.argv) > 1 else "dispatch"
RELOAD, SERVICE, STEP = 10.0, 5.0, 10
BELIEFS = os.environ.get("BELIEFS_PATH", "/app/beliefs.json")

def d(cmd):
    out = subprocess.run([DISPATCH, *cmd], capture_output=True, text=True).stdout
    return json.loads(out)

# ---------------------------------------------------------------- mode learning
class ModeLearner:
    def __init__(self, labels):
        self.labels = list(labels)
        self.samples = {m: [] for m in self.labels}     # (speed, shelf_rate)
        self.prev = None

    def observe(self, state, wait_result):
        """Turn the last interval into a (speed, shelf_rate) sample per van. Stops are allowed:
        the path is rebuilt through the orders the van served, and their 5-minute service time
        is subtracted. Intervals with a depot reload or a wait for a delivery window are skipped."""
        if not self.prev:
            return
        t0, t1 = self.prev["t"], state["t"]
        dt = t1 - t0
        before = {v["id"]: v for v in self.prev["vehicles"]}
        where = {o["id"]: o for o in state["orders"]}
        by_van = {}
        for x in wait_result.get("deliveries", []):
            by_van.setdefault(x["vehicle"], []).append(x)
        for v in state["vehicles"]:
            p = before.get(v["id"])
            if not p or v["status"] != "ok" or not p["mode"] or p["mode"] != v["mode"] or dt <= 0:
                continue
            if v["load"] > p["load"]:                       # reloaded: cargo temperature reset
                continue
            busy = max(0.0, min(p["busy_until"], t1) - t0)  # service carried over from before
            path, x, y, clean = 0.0, p["x"], p["y"], True
            for d in sorted(by_van.get(v["id"], []), key=lambda d: d["t"]):
                o = where.get(d["order"])
                if not o or abs(d["t"] - o["tw"][0]) < 1e-6:  # waited for the window: unclear
                    clean = False; break
                path += math.hypot(o["x"] - x, o["y"] - y); x, y = o["x"], o["y"]
                busy += max(0.0, min(d["t"] + SERVICE, t1) - max(d["t"], t0))
            path += math.hypot(v["x"] - x, v["y"] - y)
            drive = dt - busy
            if not clean or drive < 2 or path < 2:
                continue
            speed = path / drive
            shelf_rate = -10.0 * (v["cargo_temp_c"] - p["cargo_temp_c"]) / drive
            self.samples[p["mode"]].append((speed, shelf_rate))

    def snapshot(self, state):
        """Call right after routes/modes are set, so the next interval is attributed correctly."""
        self.prev = state

    def done(self, min_samples=2):
        return all(len(s) >= min_samples for s in self.samples.values())

    def estimate(self, m):
        s = self.samples[m]
        if not s:
            return {"speed": 1.0, "shelf_rate": 0.0}
        return {"speed": sum(a for a, _ in s) / len(s), "shelf_rate": sum(b for _, b in s) / len(s)}

    def best(self):
        est = {m: self.estimate(m) for m in self.labels if self.samples[m]}
        safe = [m for m, e in est.items() if e["shelf_rate"] >= -0.1] or list(est) or self.labels[:1]
        return max(safe, key=lambda m: self.estimate(m)["speed"])

    def beliefs(self):
        out = {}
        for m in self.labels:
            e = self.estimate(m)
            sp, sr = e["speed"], e["shelf_rate"]
            out[m] = {"speed": "faster" if sp > 1.05 else ("slower" if sp < 0.95 else "same"),
                      "cargo": "cooler" if sr > 0.05 else ("warmer" if sr < -0.05 else "same")}
        return out

# ---------------------------------------------------------------- planning
def with_reloads(seq, orders, load, cap):
    """Insert DEPOT where the load would go negative. Returns stops or None if one order > cap."""
    stops = []
    for oid in seq:
        dem = orders[oid]["demand"]
        if dem > cap:
            return None
        if dem > load:
            stops.append("DEPOT"); load = cap
        stops.append(oid); load -= dem
    return stops

def evaluate(v, t, stops, orders, mode=None):
    """Simulate a route; return (value, distance). value: on-time 1, late 0.5, spoiled 0."""
    mode = mode or {"speed": 1.0, "shelf_rate": 0.0}
    x, y, now = v["x"], v["y"], max(t, v["busy_until"])
    shelf = -10.0 * (v.get("cargo_temp_c", 4.0) - 4.0)
    value = dist = 0.0
    for s in stops:
        tx, ty = (0.0, 0.0) if s == "DEPOT" else (orders[s]["x"], orders[s]["y"])
        leg = math.hypot(tx - x, ty - y)
        dt = leg / mode["speed"]
        x, y, now, dist = tx, ty, now + dt, dist + leg
        shelf += mode["shelf_rate"] * dt
        if s == "DEPOT":
            now += RELOAD; shelf = 0.0; continue
        o = orders[s]
        start = max(now, o["tw"][0])
        if start <= o["spoil_at"] + shelf:
            value += 1.0 if start <= o["tw"][1] else 0.5
        now = start + SERVICE
    return value, dist

def plan(state, call=None, allow_loss=False, modes=None):
    """modes: optional {vehicle_id: (label, estimate)} for shifts with driving modes."""
    call = call or d
    t = state["t"]
    orders = {o["id"]: o for o in state["orders"] if o["status"] == "pending"}
    vehicles = [v for v in state["vehicles"] if v.get("status", "ok") == "ok"]
    est = {v["id"]: (modes or {}).get(v["id"], (None, None))[1] for v in vehicles}
    seqs = {v["id"]: [] for v in vehicles}
    for oid in sorted(orders, key=lambda i: orders[i]["spoil_at"]):
        best = None
        for v in vehicles:
            base = with_reloads(seqs[v["id"]], orders, v["load"], v["cap"])
            b_val, b_dist = evaluate(v, t, base, orders, est[v["id"]])
            for pos in range(len(seqs[v["id"]]) + 1):
                cand = seqs[v["id"]][:pos] + [oid] + seqs[v["id"]][pos:]
                stops = with_reloads(cand, orders, v["load"], v["cap"])
                if stops is None:
                    continue
                val, dist = evaluate(v, t, stops, orders, est[v["id"]])
                gain = (val - b_val) * 1000 - (dist - b_dist)
                if (allow_loss or val - b_val > 0) and (best is None or gain > best[0]):
                    best = (gain, v["id"], cand)
        if best:
            seqs[best[1]] = best[2]
    for v in vehicles:
        stops = with_reloads(seqs[v["id"]], orders, v["load"], v["cap"]) or []
        label = (modes or {}).get(v["id"], (None, None))[0]
        if stops != v["route"] or (label and label != v.get("mode")):
            call(["route", v["id"], *(["--mode", label] if label else []), *stops])

def mode_plan(state, learner, rnd):
    """During probing rotate labels across vans; afterwards everyone drives the best mode."""
    if learner.done() or state["t"] >= 120:
        b = learner.best()
        return {v["id"]: (b, learner.estimate(b)) for v in state["vehicles"]}
    n = len(learner.labels)
    out = {}
    for i, v in enumerate(state["vehicles"]):
        m = learner.labels[(i + rnd) % n]
        if len(learner.samples[m]) >= 2:           # this label is known; probe another one
            m = min(learner.labels, key=lambda x: len(learner.samples[x]))
        out[v["id"]] = (m, learner.estimate(m))
    return out

def run(state_fn, wait_fn, call, finish_fn, write_beliefs=None):
    s = state_fn()
    learner = ModeLearner(s["modes"]) if "modes" in s else None
    rnd, r = 0, {}
    while not s.get("error") and s["t"] < s["horizon"]:
        if learner:
            learner.observe(s, r)
            plan(s, call, modes=mode_plan(s, learner, rnd)); rnd += 1
            learner.snapshot(state_fn())
        else:
            plan(s, call)
        r = wait_fn(STEP)
        if r.get("done") or r.get("error"):
            break
        s = state_fn()
    if learner and write_beliefs:
        write_beliefs(learner.beliefs())
    finish_fn()
    return learner.beliefs() if learner else None

def main():
    def write(b):
        try:
            with open(BELIEFS, "w") as f:
                json.dump(b, f, indent=1)
        except OSError:
            pass
    run(lambda: d(["state"]), lambda m: d(["wait", str(m)]), d, lambda: d(["finish"]), write)

if __name__ == "__main__":
    main()
