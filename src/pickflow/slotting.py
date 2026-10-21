"""Storage-location assignment policies. Each returns assign[sku] -> location index.
Frequencies and affinities are learned from a *history* of orders (no leakage)."""
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def incidence(history, n_skus):
    X = np.zeros((len(history), n_skus))
    for i, (s, _) in enumerate(history):
        X[i, s] = 1
    return X


def _aisle_fill(sku_order, wh):
    """Fill aisles nearest-first, front-to-back inside each aisle."""
    locs = np.lexsort((wh.loc_side, wh.loc_y, wh.loc_aisle))
    assign = np.empty(len(sku_order), dtype=int)
    assign[sku_order] = locs[: len(sku_order)]
    return assign


def random_slotting(n_skus, wh, rng):
    return rng.permutation(wh.n_locs)[:n_skus]


def popularity_slotting(freq, wh):
    """Classic: hottest SKUs on the slots closest to the depot (scattered across aisles)."""
    locs = np.argsort(wh.D[0, 1:], kind="stable")
    assign = np.empty(len(freq), dtype=int)
    assign[np.argsort(-freq, kind="stable")] = locs[: len(freq)]
    return assign


def popularity_aisle_slotting(freq, wh):
    """Hottest SKUs concentrated in the aisles nearest the depot."""
    return _aisle_fill(np.argsort(-freq, kind="stable"), wh)


def affinity_slotting(X, freq, wh, n_clusters=24, class_cuts=(0.2, 0.5)):
    """Hybrid ABC + affinity. SKUs are split into popularity classes (A/B/C by cumulative
    SKU share); inside each class, SKUs are grouped by co-occurrence clusters (cosine
    distance, average linkage, clusters ranked by mean demand). Classes fill aisles
    nearest-first, so hot SKUs stay close while co-picked SKUs share aisles."""
    n = len(freq)
    Xn = X / np.maximum(np.linalg.norm(X, axis=0), 1e-9)
    Dm = 1.0 - Xn.T @ Xn
    np.fill_diagonal(Dm, 0.0)
    cl = fcluster(linkage(squareform(Dm, checks=False), "average"), n_clusters, "maxclust")
    rank = np.empty(n, dtype=int)
    rank[np.argsort(-freq, kind="stable")] = np.arange(n)
    cls = np.digitize(rank / n, class_cuts)            # 0 = A, 1 = B, 2 = C
    order = []
    for c in range(len(class_cuts) + 1):
        idx = np.flatnonzero(cls == c)
        mf = {k: freq[idx[cl[idx] == k]].mean() for k in np.unique(cl[idx])}
        idx = sorted(idx, key=lambda i: (-mf[cl[i]], cl[i], -freq[i]))
        order += idx
    return _aisle_fill(np.array(order), wh)
