"""
Cold-chain dynamic VRPTW environment — hours 0-6 starter (Junction 2026, AfterQuery "Break the Agent").

Pieces:
  1. Instance generator   (seeded, 3 difficulty levels)
  2. Simulator            (continuous time; vehicles execute routes only when the clock advances)
  3. Tool interface       (what the agent can call; JSON in, JSON out)
  4. Reward               (computed ONLY from the simulator's own delivery log at episode end)

Run `python coldchain_env.py` to play a greedy baseline on every level.
"""
from __future__ import annotations
import json, math, random
from dataclasses import dataclass, field

DEPOT = "DEPOT"

# ---------------------------------------------------------------- 1. instances
LEVELS = {
    #        orders  vehicles  capacity  dynamic_share  window_width  shelf_life  horizon
    "easy":   dict(n=12, k=2, cap=40, dyn=0.25, win=120, shelf=200, horizon=480),
    "medium": dict(n=25, k=3, cap=40, dyn=0.40, win=80,  shelf=150, horizon=480),
    "hard":   dict(n=40, k=4, cap=35, dyn=0.55, win=50,  shelf=110, horizon=480),
}

@dataclass
class Order:
    id: str
    x: float; y: float
    demand: int
    release: float      # when the order becomes visible to the agent
    tw_open: float      # earliest delivery
    tw_close: float     # latest on-time delivery
    spoil_at: float     # cold chain: after this the goods are worthless
    service: float = 5.0
    status: str = "pending"     # pending | delivered | late | spoiled
    delivered_at: float | None = None

def make_instance(level: str, seed: int) -> dict:
    p, rng = LEVELS[level], random.Random(seed)
    orders = []
    for i in range(p["n"]):
        dynamic = rng.random() < p["dyn"]
        release = rng.uniform(30, p["horizon"] * 0.6) if dynamic else 0.0
        x, y = rng.uniform(-50, 50), rng.uniform(-50, 50)
        earliest = release + math.hypot(x, y)          # always reachable in principle
        tw_open = earliest + rng.uniform(0, 60)
        tw_close = tw_open + p["win"]
        spoil_at = release + p["shelf"] + rng.uniform(0, 40)
        orders.append(Order(f"o{i}", round(x, 1), round(y, 1), rng.randint(3, 12),
                            round(release, 1), round(tw_open, 1), round(tw_close, 1),
                            round(spoil_at, 1)))
    return dict(level=level, seed=seed, orders=orders, k=p["k"], cap=p["cap"], horizon=p["horizon"])

# ---------------------------------------------------------------- 2. simulator
@dataclass
class Vehicle:
    id: str
    cap: int
    x: float = 0.0; y: float = 0.0
    load: int = 0
    route: list = field(default_factory=list)   # remaining stops: order ids or DEPOT
    busy_until: float = 0.0                      # serving / waiting for a window
    distance: float = 0.0

class ColdChainEnv:
    def __init__(self, level="medium", seed=0, max_tool_calls=1500):
        inst = make_instance(level, seed)
        self.level, self.seed = level, seed
        self.orders = {o.id: o for o in inst["orders"]}
        self.vehicles = {f"v{i}": Vehicle(f"v{i}", inst["cap"], load=inst["cap"]) for i in range(inst["k"])}
        self.horizon, self.t = inst["horizon"], 0.0
        self.calls, self.max_calls, self.done = 0, max_tool_calls, False
        self.log: list[dict] = []          # ground truth for reward — agent cannot write to it

    # -- visibility: the agent only sees released orders
    def _visible(self):
        return [o for o in self.orders.values() if o.release <= self.t]

    def _assigned(self):
        return {s for v in self.vehicles.values() for s in v.route if s != DEPOT}

    def _check_route(self, v: Vehicle, route: list) -> str | None:
        load, seen = v.load, set()
        for s in route:
            if s == DEPOT:
                load = v.cap; continue
            o = self.orders.get(s)
            if o is None or o.release > self.t:
                return f"unknown order {s}"
            if o.status != "pending":
                return f"order {s} already {o.status}"
            if s in seen:
                return f"order {s} listed twice"
            seen.add(s)
            load -= o.demand
            if load < 0:
                return f"capacity exceeded at {s}; insert {DEPOT} to reload"
        return None

    def _advance_vehicle(self, v: Vehicle, until: float):
        now = self.t
        while now < until:
            if v.busy_until > now:                       # serving or waiting for window
                now = min(v.busy_until, until); continue
            if not v.route:
                return
            tgt = v.route[0]
            tx, ty = (0.0, 0.0) if tgt == DEPOT else (self.orders[tgt].x, self.orders[tgt].y)
            d = math.hypot(tx - v.x, ty - v.y)
            if now + d > until:                          # partial move (speed = 1 unit/min)
                f = (until - now) / d if d else 1
                v.x += (tx - v.x) * f; v.y += (ty - v.y) * f
                v.distance += until - now
                return
            v.x, v.y, v.distance, now = tx, ty, v.distance + d, now + d
            v.route.pop(0)
            if tgt == DEPOT:
                v.load = v.cap; v.busy_until = now + 10   # reload time
                continue
            o = self.orders[tgt]
            start = max(now, o.tw_open)                   # early: wait for window
            o.delivered_at = start
            o.status = "spoiled" if start > o.spoil_at else ("late" if start > o.tw_close else "delivered")
            v.load -= o.demand
            v.busy_until = start + o.service
            self.log.append(dict(order=o.id, vehicle=v.id, t=round(start, 1), status=o.status))

    def _advance(self, minutes: float):
        until = min(self.t + minutes, self.horizon)
        for v in self.vehicles.values():
            self._advance_vehicle(v, until)
        self.t = until
        # orders past spoilage that nobody delivered are lost
        for o in self.orders.values():
            if o.status == "pending" and self.t > o.spoil_at:
                o.status = "spoiled"
        if self.t >= self.horizon:
            self.done = True

    # ------------------------------------------------------------ 3. tools
    TOOLS = [
        {"name": "get_state", "description": "Current time, vehicles (position, load, remaining route) and all visible orders.",
         "parameters": {"type": "object", "properties": {}}},
        {"name": "set_route", "description": f"Replace a vehicle's remaining stops. Stops are order ids or '{DEPOT}' (reload). Validated for capacity.",
         "parameters": {"type": "object", "properties": {"vehicle": {"type": "string"},
                        "stops": {"type": "array", "items": {"type": "string"}}}, "required": ["vehicle", "stops"]}},
        {"name": "wait", "description": "Advance the clock. Vehicles drive their routes; new orders may appear.",
         "parameters": {"type": "object", "properties": {"minutes": {"type": "number", "minimum": 1, "maximum": 120}},
                        "required": ["minutes"]}},
        {"name": "finish", "description": "End the shift now. Undelivered orders count as missed.",
         "parameters": {"type": "object", "properties": {}}},
    ]

    def step(self, tool: str, args: dict | None = None) -> dict:
        args = args or {}
        if self.done:
            return {"error": "episode over", "done": True}
        self.calls += 1
        if self.calls > self.max_calls:                  # anti-stalling: budget runs out -> shift ends
            self._advance(self.horizon); return {"done": True, "note": "tool budget exhausted"}
        if tool == "get_state":
            return self.observe()
        if tool == "set_route":
            v = self.vehicles.get(args.get("vehicle"))
            if v is None:
                return {"error": "unknown vehicle"}
            stops = list(args.get("stops", []))
            if (err := self._check_route(v, stops)):
                return {"error": err}
            v.route = stops
            return {"ok": True}
        if tool == "wait":
            m = float(args.get("minutes", 0))
            if not 1 <= m <= 120:
                return {"error": "minutes must be 1..120"}
            before = {o.id for o in self._visible()}
            self._advance(m)
            new = [o.id for o in self._visible() if o.id not in before]
            return {"t": round(self.t, 1), "new_orders": new, "done": self.done}
        if tool == "finish":
            self._advance(self.horizon)                  # time still runs out; no early-exit loophole
            return {"done": True}
        return {"error": f"unknown tool {tool}"}

    def observe(self) -> dict:
        assigned = self._assigned()
        return {
            "t": round(self.t, 1), "horizon": self.horizon,
            "vehicles": [dict(id=v.id, x=round(v.x, 1), y=round(v.y, 1), load=v.load, cap=v.cap,
                              route=v.route, busy_until=round(v.busy_until, 1)) for v in self.vehicles.values()],
            "orders": [dict(id=o.id, x=o.x, y=o.y, demand=o.demand, tw=[o.tw_open, o.tw_close],
                            spoil_at=o.spoil_at, status=o.status, assigned=o.id in assigned)
                       for o in self._visible()],
        }

    # ------------------------------------------------------------ 4. reward
    def reward(self) -> dict:
        """Terminal only. Built from the log + true order set, never from agent claims."""
        assert self.done, "reward is only defined at episode end"
        n = len(self.orders)                     # ALL orders, including ones the agent never looked at
        on_time = sum(o.status == "delivered" for o in self.orders.values())
        late = sum(o.status == "late" for o in self.orders.values())
        dist = sum(v.distance for v in self.vehicles.values())
        service = (on_time + 0.5 * late) / n     # 0..1
        score = service * (1 - min(0.3, 0.0001 * dist))  # distance discounts up to 30%; idling can never beat delivering
        return dict(score=round(score, 4), service=round(service, 3), on_time=on_time, late=late,
                    missed=n - on_time - late, distance=round(dist, 1))

# ---------------------------------------------------------------- sanity agents
def greedy_agent(env: ColdChainEnv):
    """Nearest-unassigned-order, tightest deadline first. Deliberately myopic."""
    while not env.done:
        s = env.step("get_state")
        free = [o for o in s["orders"] if o["status"] == "pending" and not o["assigned"]]
        for v in s["vehicles"]:
            if v["route"] or not free:
                continue
            free.sort(key=lambda o: (o["spoil_at"], math.hypot(o["x"] - v["x"], o["y"] - v["y"])))
            stops, load = [], v["load"]
            for o in list(free):
                if o["demand"] <= load and len(stops) < 4:
                    stops.append(o["id"]); load -= o["demand"]; free.remove(o)
            if not stops:
                stops = [DEPOT]
            env.step("set_route", {"vehicle": v["id"], "stops": stops})
        env.step("wait", {"minutes": 15})
    return env.reward()

def idle_agent(env: ColdChainEnv):
    """Reward-hack probe: do nothing. Must score ~0."""
    env.step("finish"); return env.reward()

if __name__ == "__main__":
    for lvl in LEVELS:
        g = [greedy_agent(ColdChainEnv(lvl, s))["score"] for s in range(5)]
        i = idle_agent(ColdChainEnv(lvl, 0))["score"]
        print(f"{lvl:7s} greedy mean={sum(g)/len(g):.3f}  {g}   idle={i}")
    print(json.dumps(ColdChainEnv("easy", 0).observe(), indent=1)[:600])
