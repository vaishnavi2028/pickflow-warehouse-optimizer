"""Warehouse geometry and exact aisle-based distances.

Layout: parallel picking aisles with shelves on both sides, one front and one
back cross-aisle, and a single depot (I/O point) at the front-left corner.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Warehouse:
    n_aisles: int = 10
    slots_per_side: int = 20
    aisle_spacing: float = 5.0   # metres between aisle centre-lines
    slot_width: float = 1.2      # metres per storage slot along the aisle

    def __post_init__(self):
        S = self.slots_per_side
        self.L = S * self.slot_width                      # aisle length
        self.n_locs = self.n_aisles * 2 * S
        self.loc_aisle = np.repeat(np.arange(self.n_aisles), 2 * S)
        self.loc_side = np.tile(np.repeat([0, 1], S), self.n_aisles)
        pos = np.tile(np.tile(np.arange(S), 2), self.n_aisles)
        self.loc_y = (pos + 0.5) * self.slot_width
        self.aisle_x = (np.arange(self.n_aisles) + 1) * self.aisle_spacing
        self.loc_x = self.aisle_x[self.loc_aisle]
        # node 0 = depot, node j+1 = storage location j
        self.X = np.concatenate([[0.0], self.loc_x])
        self.Y = np.concatenate([[0.0], self.loc_y])
        self.A = np.concatenate([[-1], self.loc_aisle])
        self.D = self._distance_matrix()

    def _distance_matrix(self):
        x, y, a, L = self.X, self.Y, self.A, self.L
        dx = np.abs(x[:, None] - x[None, :])
        via = np.minimum(y[:, None] + y[None, :], 2 * L - y[:, None] - y[None, :])
        same = a[:, None] == a[None, :]
        return np.where(same, np.abs(y[:, None] - y[None, :]), dx + via)
