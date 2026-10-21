"""Picker routing: closed-form S-shape heuristic and near-optimal TSP (OR-Tools)."""
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


def s_shape_cost(m, wh):
    """S-shape route length from per-aisle farthest-pick vector m (-1 = aisle not visited)."""
    idx = np.flatnonzero(m >= 0)
    k = idx.size
    if k == 0:
        return 0.0
    last = idx[-1]
    vertical = k * wh.L if k % 2 == 0 else (k - 1) * wh.L + 2 * m[last]
    return 2 * wh.aisle_x[last] + vertical


def tsp_route(nodes, wh, time_ms=150):
    """Shortest tour depot -> all `nodes` (D-indices) -> depot. Returns (length, ordered nodes)."""
    nodes = [0] + sorted(set(int(n) for n in nodes if n != 0))
    if len(nodes) == 1:
        return 0.0, [0]
    Dsub = np.rint(wh.D[np.ix_(nodes, nodes)] * 100).astype(int)
    mgr = pywrapcp.RoutingIndexManager(len(nodes), 1, 0)
    r = pywrapcp.RoutingModel(mgr)
    cb = r.RegisterTransitCallback(lambda i, j: int(Dsub[mgr.IndexToNode(i), mgr.IndexToNode(j)]))
    r.SetArcCostEvaluatorOfAllVehicles(cb)
    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.FromMilliseconds(time_ms)
    sol = r.SolveWithParameters(p)
    i, order = r.Start(0), []
    while not r.IsEnd(i):
        order.append(nodes[mgr.IndexToNode(i)])
        i = sol.Value(r.NextVar(i))
    order.append(0)
    return sol.ObjectiveValue() / 100.0, order


def route_polyline(order, wh):
    """Turn an ordered node list into x,y points that follow aisles/cross-aisles."""
    pts = []
    for i, j in zip(order[:-1], order[1:]):
        xi, yi, xj, yj = wh.X[i], wh.Y[i], wh.X[j], wh.Y[j]
        if wh.A[i] == wh.A[j] and i != 0:
            pts += [(xi, yi), (xj, yj)]
        else:
            yc = 0.0 if yi + yj <= 2 * wh.L - yi - yj else wh.L
            pts += [(xi, yi), (xi, yc), (xj, yc), (xj, yj)]
    return np.array(pts)
