"""Scenario evaluation: distance, labour time, and multi-picker makespan."""
import numpy as np
from .batching import order_vectors, fcfs, savings, local_search, _vec
from .routing import s_shape_cost, tsp_route
from . import slotting as sl

SPEED = 1.0        # m/s walking with cart
PICK_S = 6.0       # s per item picked
SETUP_S = 90.0     # s per batch (cart prep + hand-in)


def build_assignment(method, wh, history, n_skus, rng, n_clusters=24):
    X = sl.incidence(history, n_skus)
    freq = X.sum(axis=0) + 1e-6 * rng.random(n_skus)   # tiny noise breaks ties for unseen SKUs
    return {
        "random": lambda: sl.random_slotting(n_skus, wh, rng),
        "popularity": lambda: sl.popularity_slotting(freq, wh),
        "popularity_aisle": lambda: sl.popularity_aisle_slotting(freq, wh),
        "affinity": lambda: sl.affinity_slotting(X, freq, wh, n_clusters),
    }[method]()


def make_batches(method, orders, assign, wh, cap):
    A = order_vectors(orders, assign, wh)
    q = np.array([int(qty.sum()) for _, qty in orders])
    if method == "fcfs":
        b = fcfs(q, cap)
    elif method == "savings":
        b = savings(A, q, cap, wh)
    elif method == "savings+ls":
        b = local_search(savings(A, q, cap, wh), A, q, cap, wh)
    else:
        raise ValueError(method)
    return b, A, q


def evaluate(batches, A, q, wh, cap, n_pickers=4):
    dist = np.array([s_shape_cost(_vec(A, b, wh), wh) for b in batches])
    items = np.array([q[b].sum() for b in batches])
    t = dist / SPEED + items * PICK_S + SETUP_S
    loads = np.zeros(n_pickers)
    for x in sorted(t, reverse=True):                  # LPT schedule across pickers
        loads[loads.argmin()] += x
    return dict(batches=len(batches), distance_m=dist.sum(),
                dist_per_order_m=dist.sum() / len(q), dist_per_item_m=dist.sum() / q.sum(),
                labour_h=t.sum() / 3600, makespan_h=loads.max() / 3600,
                avg_fill=items.mean() / cap)


def tsp_total(batches, orders, assign, wh, time_ms=150):
    total, routes = 0.0, []
    for b in batches:
        nodes = np.concatenate([assign[orders[o][0]] for o in b]) + 1
        d, r = tsp_route(nodes, wh, time_ms)
        total += d; routes.append(r)
    return total, routes
