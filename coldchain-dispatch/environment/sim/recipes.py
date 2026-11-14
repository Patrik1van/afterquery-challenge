"""Scenario recipes: each one is built to trigger a specific, named failure mode of LLM agents.

Every recipe is seeded and solvable: the reference dispatcher (solution/reference.py) scores well
on it. What changes is *which* habit gets punished. Map: depot at (0,0), speed 1 unit/min,
shift = 480 min, service = 5 min, depot reload = 10 min.

recipe            failure mode it targets
----------------  ---------------------------------------------------------------------------
nearest_trap      greedy "closest first": a juicy cluster by the depot hides far, urgent orders
short_fuse        coarse polling: orders that spoil ~30 min after they appear punish long waits
rush_hour         no replanning: a surge of orders lands mid-shift on top of committed routes
waiting_game      ignoring time windows: nearby orders that open late make vans idle for hours
heavy_load        capacity arithmetic: big orders need depot reloads planned into routes
breakdown         not reading state changes: a van breaks down mid-shift, its stops need rescue
overbooked        no triage: more work than the fleet can do; trying to serve all spoils more
*_hidden          no hypothesis testing: vans have unlabeled driving modes with hidden effects on
                  speed and cargo temperature; the agent must experiment, infer, then exploit
"""
from __future__ import annotations
import math, random

H = 480

def _polar(rng, rmin, rmax):
    a, r = rng.uniform(0, 2 * math.pi), rng.uniform(rmin, rmax)
    return round(r * math.cos(a), 1), round(r * math.sin(a), 1)

def _order(Order, i, x, y, demand, release, tw_open, tw_close, spoil_at):
    return Order(f"o{i}", x, y, demand, round(release, 1), round(tw_open, 1),
                 round(tw_close, 1), round(spoil_at, 1))

def nearest_trap(rng, Order):
    orders, i = [], 0
    for _ in range(12):                                   # decoy cluster: close, relaxed
        x, y = _polar(rng, 3, 14)
        orders.append(_order(Order, i, x, y, rng.randint(2, 5), 0, 0, H, H)); i += 1
    for k in range(6):                                    # urgent and far, in two opposite arcs
        ang = (0 if k % 2 else math.pi) + rng.uniform(-0.6, 0.6)
        r = rng.uniform(35, 48)
        x, y = round(r * math.cos(ang), 1), round(r * math.sin(ang), 1)
        d = math.hypot(x, y)
        orders.append(_order(Order, i, x, y, rng.randint(3, 6), 0, 0, d + rng.uniform(35, 55),
                             d + rng.uniform(60, 75))); i += 1
    rng.shuffle(orders)
    return dict(orders=orders, k=2, cap=40, events=[])

def short_fuse(rng, Order):
    orders, i = [], 0
    for _ in range(10):                                   # background work with wide windows
        x, y = _polar(rng, 10, 40)
        orders.append(_order(Order, i, x, y, rng.randint(3, 8), 0, 0, 300, 420)); i += 1
    for _ in range(12):                                   # appear, then spoil fast
        rel = rng.uniform(40, 400)
        x, y = _polar(rng, 6, 22)
        d = math.hypot(x, y)
        orders.append(_order(Order, i, x, y, rng.randint(2, 5), rel, rel, rel + d + 20,
                             rel + d + rng.uniform(28, 40))); i += 1
    return dict(orders=orders, k=3, cap=40, events=[])

def rush_hour(rng, Order):
    orders, i = [], 0
    for _ in range(8):
        x, y = _polar(rng, 10, 45)
        orders.append(_order(Order, i, x, y, rng.randint(3, 8), 0, 0, 260, 360)); i += 1
    for _ in range(20):                                   # surge between minute 200 and 230
        rel = rng.uniform(200, 230)
        x, y = _polar(rng, 8, 45)
        d = math.hypot(x, y)
        op = rel + d + rng.uniform(0, 30)
        orders.append(_order(Order, i, x, y, rng.randint(2, 6), rel, op, op + rng.uniform(50, 90),
                             op + rng.uniform(100, 140))); i += 1
    return dict(orders=orders, k=4, cap=40, events=[])

def waiting_game(rng, Order):
    orders, i = [], 0
    for _ in range(10):                                   # right next to the depot, but open late
        x, y = _polar(rng, 4, 15)
        op = rng.uniform(220, 320)
        orders.append(_order(Order, i, x, y, rng.randint(2, 5), 0, op, op + 60, op + 110)); i += 1
    for _ in range(12):                                   # farther, open now, spoil by ~200
        x, y = _polar(rng, 20, 45)
        d = math.hypot(x, y)
        orders.append(_order(Order, i, x, y, rng.randint(3, 7), 0, 0, d + rng.uniform(60, 120),
                             d + rng.uniform(140, 170))); i += 1
    rng.shuffle(orders)
    return dict(orders=orders, k=2, cap=45, events=[])

def heavy_load(rng, Order):
    orders = []
    for i in range(18):
        x, y = _polar(rng, 10, 40)
        d = math.hypot(x, y)
        op = rng.uniform(0, 200)
        orders.append(_order(Order, i, x, y, rng.randint(9, 15), 0, op, op + d + 120, op + d + 200))
    return dict(orders=orders, k=2, cap=30, events=[])

def breakdown(rng, Order):
    orders = []
    for i in range(22):
        dyn = i >= 14
        rel = rng.uniform(60, 250) if dyn else 0
        x, y = _polar(rng, 10, 45)
        d = math.hypot(x, y)
        op = rel + d + rng.uniform(0, 40)
        orders.append(_order(Order, i, x, y, rng.randint(3, 7), rel, op, op + rng.uniform(70, 110),
                             op + rng.uniform(130, 170)))
    return dict(orders=orders, k=3, cap=35, events=[dict(t=130.0, type="breakdown", vehicle="v0")])

def overbooked(rng, Order):
    orders = []
    for i in range(34):
        rel = 0 if i < 20 else rng.uniform(30, 300)
        x, y = _polar(rng, 10, 50)
        d = math.hypot(x, y)
        op = rel + d + rng.uniform(0, 80)
        cl = op + rng.uniform(25, 45)                     # late barely helps: spoils soon after close
        orders.append(_order(Order, i, x, y, rng.randint(3, 8), rel, op, cl, cl + rng.uniform(3, 8)))
    return dict(orders=orders, k=2, cap=40, events=[])

# ---- hidden-rule layer: unlabeled driving modes ---------------------------------------------
# Like operators in an equation puzzle: the labels are fixed for the shift, the meaning behind each
# label is shuffled per seed, and the agent has to infer it from evidence that costs shift time.
MODE_FAMILY = [
    {"speed": 1.0, "shelf_rate": 0.0},    # plain
    {"speed": 1.6, "shelf_rate": -0.6},   # express: fast, but the reefer can't keep up
    {"speed": 0.7, "shelf_rate": 0.5},    # eco-cold: slow, cargo lasts longer
    {"speed": 0.8, "shelf_rate": -1.2},   # lemon: slower AND cooks the cargo
]
MODE_LABELS = ["amber", "cobalt", "jade", "slate"]

def with_hidden_modes(base):
    def recipe(rng, Order):
        inst = base(rng, Order)
        fam = MODE_FAMILY[:]; rng.shuffle(fam)
        inst["modes"] = dict(zip(MODE_LABELS, fam))
        return inst
    recipe.__name__ = base.__name__ + "_hidden"
    return recipe

RECIPES = {f.__name__: f for f in
           [nearest_trap, short_fuse, rush_hour, waiting_game, heavy_load, breakdown, overbooked,
            with_hidden_modes(nearest_trap), with_hidden_modes(short_fuse)]}

def build(name, seed, Order):
    inst = RECIPES[name](random.Random(f"{name}:{seed}"), Order)
    inst.update(level=name, seed=seed, horizon=H)
    return inst
