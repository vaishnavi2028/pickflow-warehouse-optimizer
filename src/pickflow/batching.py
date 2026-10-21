"""Order batching: FCFS baseline, Clarke-Wright savings, and local-search improvement.
Cost of a batch = S-shape route length, computed from per-aisle farthest-pick vectors."""
import heapq
import numpy as np
from .routing import s_shape_cost


def order_vectors(orders, assign, wh):
    """A[o, a] = farthest pick (y) of order o in aisle a, -1 if the aisle is not visited."""
    A = np.full((len(orders), wh.n_aisles), -1.0)
    for o, (s, _) in enumerate(orders):
        locs = assign[s]
        np.maximum.at(A[o], wh.loc_aisle[locs], wh.loc_y[locs])
    return A


def _vec(A, members, wh):
    return A[members].max(axis=0) if len(members) else np.full(wh.n_aisles, -1.0)


def total_cost(batches, A, wh):
    return sum(s_shape_cost(_vec(A, b, wh), wh) for b in batches)


def fcfs(q, cap):
    batches, cur, load = [], [], 0
    for o in range(len(q)):
        if load + q[o] > cap and cur:
            batches.append(cur); cur, load = [], 0
        cur.append(o); load += q[o]
    return batches + ([cur] if cur else [])


def savings(A, q, cap, wh):
    """Clarke-Wright: repeatedly merge the pair of batches with the largest route saving."""
    n = len(q)
    mem = {i: [i] for i in range(n)}
    vec = {i: A[i].copy() for i in range(n)}
    cost = {i: s_shape_cost(vec[i], wh) for i in range(n)}
    load = {i: q[i] for i in range(n)}
    heap = []

    def push(a, b):
        if load[a] + load[b] <= cap:
            s = cost[a] + cost[b] - s_shape_cost(np.maximum(vec[a], vec[b]), wh)
            if s > 1e-9:
                heapq.heappush(heap, (-s, a, b))

    for a in range(n):
        for b in range(a + 1, n):
            push(a, b)
    nxt = n
    while heap:
        _, a, b = heapq.heappop(heap)
        if a not in mem or b not in mem:
            continue
        m = nxt; nxt += 1
        mem[m] = mem.pop(a) + mem.pop(b)
        vec[m] = np.maximum(vec.pop(a), vec.pop(b))
        cost[m] = s_shape_cost(vec[m], wh)
        load[m] = load.pop(a) + load.pop(b)
        cost.pop(a); cost.pop(b)
        for c in list(mem):
            if c != m:
                push(c, m)
    return list(mem.values())


def local_search(batches, A, q, cap, wh, max_passes=4):
    """Relocate and swap moves between batches until no move reduces total route length."""
    B = [list(b) for b in batches]
    for _ in range(max_passes):
        improved = False
        # relocate
        vec = [_vec(A, b, wh) for b in B]
        cst = [s_shape_cost(v, wh) for v in vec]
        load = [int(q[b].sum()) for b in B]
        for bi in range(len(B)):
            for o in list(B[bi]):
                if o not in B[bi]:
                    continue
                rest = [x for x in B[bi] if x != o]
                gain = cst[bi] - s_shape_cost(_vec(A, rest, wh), wh)
                best, tgt = 1e-9, None
                for bj in range(len(B)):
                    if bj == bi or load[bj] + q[o] > cap:
                        continue
                    d = gain - (s_shape_cost(np.maximum(vec[bj], A[o]), wh) - cst[bj])
                    if d > best:
                        best, tgt = d, bj
                if tgt is not None:
                    B[bi] = rest; B[tgt].append(o); improved = True
                    for k in (bi, tgt):
                        vec[k] = _vec(A, B[k], wh); cst[k] = s_shape_cost(vec[k], wh)
                        load[k] = int(q[B[k]].sum()) if B[k] else 0
        # swap
        for bi in range(len(B)):
            for bj in range(bi + 1, len(B)):
                done = False
                for o in list(B[bi]):
                    for p in list(B[bj]):
                        if o not in B[bi] or p not in B[bj]:
                            continue
                        if load[bi] - q[o] + q[p] > cap or load[bj] - q[p] + q[o] > cap:
                            continue
                        ni = [x for x in B[bi] if x != o] + [p]
                        nj = [x for x in B[bj] if x != p] + [o]
                        new = s_shape_cost(_vec(A, ni, wh), wh) + s_shape_cost(_vec(A, nj, wh), wh)
                        if new < cst[bi] + cst[bj] - 1e-9:
                            B[bi], B[bj] = ni, nj; improved = True
                            for k in (bi, bj):
                                vec[k] = _vec(A, B[k], wh); cst[k] = s_shape_cost(vec[k], wh)
                                load[k] = int(q[B[k]].sum())
                            done = True
                            break
                    if done:
                        break
        B = [b for b in B if b]
        if not improved:
            break
    return B
