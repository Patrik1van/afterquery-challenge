#!/usr/bin/env python3
"""Reference dispatcher: rolling-horizon cheapest insertion.

Every 10 simulated minutes it rebuilds all routes from scratch:
  - orders sorted by spoilage deadline (most urgent first)
  - each order inserted at the vehicle/position that keeps the most deliveries on time
    and adds the least driving; DEPOT reloads are inserted automatically when needed.
It talks to the simulator only through the same `dispatch` CLI the agent uses.
"""
import json, math, subprocess, sys

DISPATCH = sys.argv[1] if len(sys.argv) > 1 else "dispatch"
RELOAD, SERVICE = 10.0, 5.0

def d(cmd):
    out = subprocess.run([DISPATCH, *cmd], capture_output=True, text=True).stdout
    return json.loads(out)

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

def evaluate(v, t, stops, orders):
    """Simulate a route; return (value, distance). value: on-time 1, late 0.5, spoiled 0."""
    x, y, now = v["x"], v["y"], max(t, v["busy_until"])
    value = dist = 0.0
    for s in stops:
        tx, ty = (0.0, 0.0) if s == "DEPOT" else (orders[s]["x"], orders[s]["y"])
        leg = math.hypot(tx - x, ty - y)
        x, y, now, dist = tx, ty, now + leg, dist + leg
        if s == "DEPOT":
            now += RELOAD; continue
        o = orders[s]
        start = max(now, o["tw"][0])
        if start <= o["spoil_at"]:
            value += 1.0 if start <= o["tw"][1] else 0.5
        now = start + SERVICE
    return value, dist

def plan(state):
    t = state["t"]
    orders = {o["id"]: o for o in state["orders"] if o["status"] == "pending"}
    vehicles = state["vehicles"]
    seqs = {v["id"]: [] for v in vehicles}
    for oid in sorted(orders, key=lambda i: orders[i]["spoil_at"]):
        best = None
        for v in vehicles:
            base = with_reloads(seqs[v["id"]], orders, v["load"], v["cap"])
            b_val, b_dist = evaluate(v, t, base, orders)
            for pos in range(len(seqs[v["id"]]) + 1):
                cand = seqs[v["id"]][:pos] + [oid] + seqs[v["id"]][pos:]
                stops = with_reloads(cand, orders, v["load"], v["cap"])
                if stops is None:
                    continue
                val, dist = evaluate(v, t, stops, orders)
                gain = (val - b_val) * 1000 - (dist - b_dist)
                if val - b_val > 0 and (best is None or gain > best[0]):
                    best = (gain, v["id"], cand)
        if best:
            seqs[best[1]] = best[2]
    for v in vehicles:
        stops = with_reloads(seqs[v["id"]], orders, v["load"], v["cap"]) or []
        if stops != v["route"]:
            d(["route", v["id"], *stops])

def main():
    while True:
        s = d(["state"])
        if s.get("error") or s["t"] >= s["horizon"]:
            break
        plan(s)
        r = d(["wait", "10"])
        if r.get("done") or r.get("error"):
            break
    d(["finish"])

if __name__ == "__main__":
    main()
