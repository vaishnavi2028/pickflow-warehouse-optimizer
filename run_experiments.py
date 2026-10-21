"""Run all experiments and write tables/figures to results/.  Usage: python run_experiments.py"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pickflow.layout import Warehouse
from pickflow.data import make_catalog, make_orders
from pickflow.evaluate import build_assignment, make_batches, evaluate, tsp_total
from pickflow.routing import route_polyline

OUT = Path(__file__).parent / "results"; OUT.mkdir(exist_ok=True)
N_SKUS, N_FAM, N_HIST, N_ORDERS, CAP, SEEDS = 360, 24, 2000, 300, 45, range(5)
SLOT = ["random", "popularity", "popularity_aisle", "affinity"]
BATCH = ["fcfs", "savings", "savings+ls"]
LABEL = {"random": "Random", "popularity": "Popularity (nearest slots)",
         "popularity_aisle": "Popularity (aisle fill)", "affinity": "ABC + affinity clusters"}
wh = Warehouse()


def instance(seed):
    rng = np.random.default_rng(seed)
    pop, fam = make_catalog(N_SKUS, N_FAM, 1.0, rng)
    return rng, make_orders(N_HIST, pop, fam, rng), make_orders(N_ORDERS, pop, fam, rng)


# 1) Full factorial: slotting x batching (S-shape routing), 5 replications ----------------
rows = []
for seed in SEEDS:
    rng, hist, orders = instance(seed)
    for s in SLOT:
        a = build_assignment(s, wh, hist, N_SKUS, rng)
        for b in BATCH:
            t0 = time.time()
            batches, A, q = make_batches(b, orders, a, wh, CAP)
            rows.append(dict(seed=seed, slotting=s, batching=b, solve_s=time.time() - t0,
                             **evaluate(batches, A, q, wh, CAP)))
    print("seed", seed, "done", flush=True)
df = pd.DataFrame(rows)
df.to_csv(OUT / "grid_results.csv", index=False)
summ = df.groupby(["slotting", "batching"])[["distance_m", "dist_per_order_m", "labour_h", "makespan_h", "avg_fill"]].mean()
base = summ.loc[("random", "fcfs"), "distance_m"]
summ["saving_vs_baseline_%"] = 100 * (1 - summ["distance_m"] / base)
summ = summ.round(2).reset_index()
summ.to_csv(OUT / "summary.csv", index=False)
print(summ.to_string(index=False))

# Fig 1: grouped bars --------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.8))
w = 0.26
for k, b in enumerate(BATCH):
    g = df[df.batching == b].groupby("slotting")["distance_m"].agg(["mean", "std"]).loc[SLOT]
    ax.bar(np.arange(len(SLOT)) + (k - 1) * w, g["mean"] / 1000, w, yerr=g["std"] / 1000, capsize=3, label=b)
ax.set_xticks(range(len(SLOT))); ax.set_xticklabels([LABEL[s] for s in SLOT], fontsize=8)
ax.set_ylabel("Total picker travel per 300 orders (km)"); ax.legend(title="Batching")
ax.set_title("Slotting x batching: total travel distance (mean ± sd, 5 seeds)")
fig.tight_layout(); fig.savefig(OUT / "fig1_scenarios.png", dpi=150); plt.close(fig)

# 2) Capacity sensitivity (seed 0..2, Savings batching) -------------------------------------
rows = []
for seed in range(3):
    rng, hist, orders = instance(seed)
    for s in ["random", "affinity"]:
        a = build_assignment(s, wh, hist, N_SKUS, rng)
        for cap in [15, 25, 35, 45, 60, 80]:
            for b in ["fcfs", "savings"]:
                batches, A, q = make_batches(b, orders, a, wh, cap)
                rows.append(dict(seed=seed, slotting=s, batching=b, cap=cap,
                                 **evaluate(batches, A, q, wh, cap)))
sens = pd.DataFrame(rows); sens.to_csv(OUT / "sensitivity_capacity.csv", index=False)
fig, ax = plt.subplots(figsize=(7, 4.5))
for (s, b), g in sens.groupby(["slotting", "batching"]):
    m = g.groupby("cap")["dist_per_order_m"].mean()
    ax.plot(m.index, m.values, marker="o", label=f"{LABEL[s]} + {b}")
ax.set_xlabel("Cart capacity (items)"); ax.set_ylabel("Travel per order (m)")
ax.set_title("Sensitivity to cart capacity"); ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "fig2_capacity.png", dpi=150); plt.close(fig)

# 3) S-shape vs near-optimal TSP re-routing, all 5 seeds -------------------------------------
# Batches are built with the fast S-shape cost, then each batch is re-routed with an OR-Tools TSP
# solve to measure how much the simple routing rule loses. (Seed 0 is also used for the figures.)
cases = {"Baseline: random + FCFS": ("random", "fcfs"), "PickFlow: affinity + savings+LS": ("affinity", "savings+ls")}
gap_rows, picks, routes_store = [], {}, {}
for seed in SEEDS:
    rng, hist, orders = instance(seed)
    for name, (s, b) in cases.items():
        a = build_assignment(s, wh, hist, N_SKUS, np.random.default_rng(seed))
        batches, A, q = make_batches(b, orders, a, wh, CAP)
        ev = evaluate(batches, A, q, wh, CAP)
        tsp_d, routes = tsp_total(batches, orders, a, wh)
        gap_rows.append(dict(seed=seed, scenario=name, sshape_m=ev["distance_m"], tsp_m=tsp_d))
        if seed == 0:
            routes_store[name] = (routes, batches, a)
            cnt = np.zeros(wh.n_locs)
            for sk, _ in orders:
                np.add.at(cnt, a[sk], 1)
            picks[name] = cnt
    print("tsp seed", seed, "done", flush=True)
gp = pd.DataFrame(gap_rows); gp.to_csv(OUT / "routing_gap_by_seed.csv", index=False)
gap = gp.groupby("scenario", sort=False)[["sshape_m", "tsp_m"]].mean()
gap["sshape_over_tsp_pct"] = 100 * (gap.sshape_m / gap.tsp_m - 1)
gap = gap.round(1).reset_index(); gap.to_csv(OUT / "routing_gap.csv", index=False)
print(gap.to_string(index=False))
piv = gp.pivot(index="seed", columns="scenario", values="tsp_m")
sv = 100 * (1 - piv["PickFlow: affinity + savings+LS"] / piv["Baseline: random + FCFS"])
print(f"End-to-end saving with TSP routing: mean {sv.mean():.1f}% (sd {sv.std():.1f}, min {sv.min():.1f}, max {sv.max():.1f})")
pd.DataFrame({"saving_pct": sv}).to_csv(OUT / "end_to_end_saving_tsp.csv")

# Fig 3: pick-density heatmap on the floor plan
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for ax, (name, cnt) in zip(axes, picks.items()):
    sc = ax.scatter(wh.loc_x + (wh.loc_side - 0.5) * 1.6, wh.loc_y, c=cnt, s=14, cmap="YlOrRd", vmax=np.percentile(cnt, 99))
    ax.plot(0, 0, "ks"); ax.set_title(name, fontsize=10); ax.set_xlabel("x (m)")
axes[0].set_ylabel("y (m)  [depot = black square]"); fig.colorbar(sc, ax=axes, label="picks per location")
fig.savefig(OUT / "fig3_pick_density.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# Fig 4: sample batch route, before vs after
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for ax, (name, (routes, batches, a)) in zip(axes, routes_store.items()):
    k = int(np.argmax([len(r) for r in routes]))
    ax.scatter(wh.loc_x + (wh.loc_side - 0.5) * 1.6, wh.loc_y, s=4, c="lightgrey")
    pts = route_polyline(routes[k], wh)
    ax.plot(pts[:, 0], pts[:, 1], "-", lw=1.6, color="tab:blue")
    pk = np.array(routes[k][1:-1]); ax.scatter(wh.X[pk] + (wh.loc_side[pk - 1] - 0.5) * 1.6, wh.Y[pk], s=22, c="tab:red", zorder=3)
    d = np.abs(np.diff(pts, axis=0)).sum()
    ax.set_title(f"{name}\nbatch of {len(batches[k])} orders, route {d:.0f} m", fontsize=10); ax.set_xlabel("x (m)")
axes[0].set_ylabel("y (m)")
fig.tight_layout(); fig.savefig(OUT / "fig4_routes.png", dpi=150); plt.close(fig)
print("done")
