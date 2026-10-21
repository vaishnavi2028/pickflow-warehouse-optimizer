"""Synthetic but realistic demand: Zipf SKU popularity + product-family affinity."""
import numpy as np


def make_catalog(n_skus, n_families, alpha, rng):
    pop = 1.0 / np.arange(1, n_skus + 1) ** alpha
    pop = rng.permutation(pop / pop.sum())            # popularity independent of SKU id
    fam = rng.permutation(np.arange(n_skus) % n_families)
    return pop, fam


def make_orders(n_orders, pop, fam, rng, mean_lines=4, p_family=0.65, max_lines=12):
    """Each order has a 'primary family'; each line comes from it w.p. p_family,
    otherwise from the whole catalogue. Returns list of (skus, qty)."""
    n_fam = fam.max() + 1
    fam_pop = np.bincount(fam, weights=pop)
    fam_pop /= fam_pop.sum()
    members = [np.flatnonzero(fam == f) for f in range(n_fam)]
    orders = []
    for _ in range(n_orders):
        k = int(np.clip(1 + rng.poisson(mean_lines - 1), 1, max_lines))
        f = rng.choice(n_fam, p=fam_pop)
        pm = pop[members[f]] / pop[members[f]].sum()
        chosen, tries = set(), 0
        while len(chosen) < k and tries < 60:
            tries += 1
            if rng.random() < p_family:
                chosen.add(int(rng.choice(members[f], p=pm)))
            else:
                chosen.add(int(rng.choice(len(pop), p=pop)))
        skus = np.array(sorted(chosen))
        qty = rng.choice([1, 2, 3], size=len(skus), p=[0.6, 0.3, 0.1])
        orders.append((skus, qty))
    return orders
