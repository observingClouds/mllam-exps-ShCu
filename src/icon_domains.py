"""Programmatic construction of sub-domains on the (LAM) ICON grid.

ICON grids are built by recursive bisection, and cells are stored in that
hierarchical order, so blocks of ``4**level`` consecutive cells usually form a
triangle of side ``2**level``. In limited-area grids the lateral boundary rows
(``refin_c_ctrl`` 1..5) are moved to the front of the cell list. That leaves
gaps in the hierarchical ordering, so fixed offsets drift near the boundary.

Instead of relying on offsets, this module rebuilds the hierarchy bottom-up:
four consecutive, index-contiguous triangles of level ``k`` are merged into a
level ``k+1`` triangle only if their corner vertices actually form one (3
vertices occur once = parent corners, 3 occur three times = edge midpoints).
Level 1 is additionally constrained to ICON's own ``parent_cell_index``.
Incomplete triangles at the boundary simply stay unmerged.

Domains are then built from complete level-``L`` triangles using vertex/edge
adjacency, e.g. the 2x2 parallelogram (8 triangles) used in ``data/domains.py``.

Note: "level" here counts factors of 4 cells, i.e. ``data/domains.py``'s
``16**4`` corresponds to ``level=8`` (4**8 = 65536 cells per triangle).

Example
-------
>>> grid = load_grid()                      # or xr.open_dataset("grid.nc")
>>> tree = TriangleTree(grid)
>>> cands = tree.parallelograms(level=8)    # all 2x2 parallelograms
>>> chosen = select_non_overlapping(cands)
>>> tree.parallelogram_at(13.3, -57.7, level=7)  # smaller domain around a point
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import xarray as xr

GRID_URL = (
    "https://swift.dkrz.de/v1/dkrz_948e7d4bbfbb445fbff5315fc433e36a/"
    "grids/EUREC4A_PR1250m_DOM02.zarr"
)
GRID_VARS = ["vertex_of_cell", "parent_cell_index", "refin_c_ctrl", "vlat", "vlon"]


def load_grid(url: str = GRID_URL, cache: str | None = None) -> xr.Dataset:
    """Load the topology variables needed here (optionally caching to netCDF)."""
    if cache is not None:
        try:
            return xr.open_dataset(cache)
        except FileNotFoundError:
            pass
    grid = xr.open_zarr(url)[GRID_VARS].load()
    if cache is not None:
        grid.to_netcdf(cache)
    return grid


def _merge_level(start, stop, corners, key=None):
    """Merge runs of 4 consecutive units that form a parent triangle (greedy, left to right)."""
    n = len(start)
    if n < 4:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros((0, 3), corners.dtype)
    j = np.arange(n - 3)
    valid = (stop[j] == start[j + 1]) & (stop[j + 1] == start[j + 2]) & (stop[j + 2] == start[j + 3])
    if key is not None:
        valid &= (key[j] == key[j + 1]) & (key[j] == key[j + 2]) & (key[j] == key[j + 3])
    s = np.sort(np.concatenate([corners[j + o] for o in range(4)], axis=1), axis=1)
    eq = s[:, 1:] == s[:, :-1]
    pad = np.zeros((len(j), 1), bool)
    single = ~np.hstack([pad, eq]) & ~np.hstack([eq, pad])
    valid &= (eq.sum(1) == 6) & (single.sum(1) == 3) & ~(s[:, 3:] == s[:, :-3]).any(1)

    chosen, last = [], -4
    for k in np.flatnonzero(valid):
        if k >= last + 4:
            chosen.append(k)
            last = k
    chosen = np.asarray(chosen, dtype=np.int64)
    return start[chosen], stop[chosen + 3], s[chosen][single[chosen]].reshape(-1, 3)


@dataclass
class Domain:
    level: int
    triangles: np.ndarray  # indices into TriangleTree.levels[level]
    slices: list[slice]
    center_vertex: int

    @property
    def n_cells(self) -> int:
        return sum(s.stop - s.start for s in self.slices)

    def cells(self) -> np.ndarray:
        return np.concatenate([np.arange(s.start, s.stop) for s in self.slices])


class TriangleTree:
    """Hierarchy of complete triangles, recovered from cell order + topology."""

    def __init__(self, grid: xr.Dataset, max_level: int = 12):
        self.vlat = np.rad2deg(grid["vlat"].values)
        self.vlon = np.rad2deg(grid["vlon"].values)
        voc = grid["vertex_of_cell"].values.astype(np.int64) - 1  # 1-based in ICON files
        ncell = voc.shape[1]
        start = np.arange(ncell, dtype=np.int64)
        stop = start + 1
        corners = voc.T.copy()
        key = grid["parent_cell_index"].values if "parent_cell_index" in grid else None
        self.levels = [dict(start=start, stop=stop, corners=corners)]
        for _ in range(max_level):
            start, stop, corners = _merge_level(start, stop, corners, key=key)
            key = None
            if len(start) == 0:
                break
            self.levels.append(dict(start=start, stop=stop, corners=corners))
        self._adj_cache: dict[int, tuple] = {}

    def summary(self) -> None:
        for lev, d in enumerate(self.levels):
            print(f"level {lev:2d}: {len(d['start']):9d} complete triangles of {4**lev} cells")

    # ------------------------------------------------------------------ adjacency
    def _adjacency(self, level: int):
        if level not in self._adj_cache:
            corners = self.levels[level]["corners"]
            by_vertex, by_edge = defaultdict(list), defaultdict(list)
            for t, (a, b, c) in enumerate(corners.tolist()):
                for v in (a, b, c):
                    by_vertex[v].append(t)
                for e in ((a, b), (b, c), (a, c)):
                    by_edge[tuple(sorted(e))].append(t)
            self._adj_cache[level] = (by_vertex, by_edge)
        return self._adj_cache[level]

    def _slices(self, level: int, tris) -> list[slice]:
        d = self.levels[level]
        ranges = sorted(zip(d["start"][tris].tolist(), d["stop"][tris].tolist()))
        merged = [list(ranges[0])]
        for a, b in ranges[1:]:
            if a == merged[-1][1]:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        return [slice(a, b) for a, b in merged]

    # ------------------------------------------------------------------ shapes
    def hexagons(self, level: int) -> list[Domain]:
        """All hexagons (6 complete triangles around one vertex)."""
        by_vertex, _ = self._adjacency(level)
        return [
            Domain(level, np.array(ts), self._slices(level, ts), v)
            for v, ts in by_vertex.items()
            if len(ts) == 6
        ]

    def domain_from_slices(self, slices: list[slice], level: int) -> Domain:
        """Domain made of the complete level-``level`` triangles covered by ``slices``
        (e.g. an entry of ``data/domains.py``)."""
        d = self.levels[level]
        covered = np.zeros(len(self.levels[0]["start"]) + 1, np.int64)
        for s in slices:
            covered[s.start] += 1
            covered[s.stop] -= 1
        inside = np.cumsum(covered)[:-1] > 0
        csum = np.r_[0, np.cumsum(inside)]
        tris = np.flatnonzero(csum[d["stop"]] - csum[d["start"]] == d["stop"] - d["start"])
        if len(tris) == 0:
            raise ValueError(f"slices contain no complete level-{level} triangle")
        u, n = np.unique(d["corners"][tris], return_counts=True)
        return Domain(level, tris, self._slices(level, tris), int(u[np.argmax(n)]))

    def orientation(self, dom: Domain) -> float:
        """Direction [deg, 0..180, counter-clockwise from east] of the long diagonal,
        i.e. the line through the two acute corners (corners used by one triangle only).
        Bisection preserves lattice directions, so this is comparable across levels.
        """
        u, n = np.unique(self.levels[dom.level]["corners"][dom.triangles], return_counts=True)
        a, b = u[n == 1]
        lat0 = np.deg2rad(self.vlat[dom.center_vertex])
        dx = (self.vlon[b] - self.vlon[a]) * np.cos(lat0)
        dy = self.vlat[b] - self.vlat[a]
        return float(np.rad2deg(np.arctan2(dy, dx)) % 180)

    def parallelograms(self, level: int, orientation: float | Domain | None = None,
                       tol: float = 20.0) -> list[Domain]:
        """All 2x2 parallelograms (8 triangles): a hexagon plus the two triangles
        beyond the outer edges of two opposite hexagon triangles. Up to 3 per vertex
        (one per orientation). ``orientation`` (deg, see :meth:`orientation`, or a
        reference Domain) keeps only those whose long diagonal is within ``tol`` deg;
        the three possible orientations are 60 deg apart.
        """
        cands = self._parallelograms(level)
        if orientation is None:
            return cands
        if isinstance(orientation, Domain):
            orientation = self.orientation(orientation)
        keep = []
        for c in cands:
            diff = abs(self.orientation(c) - orientation) % 180
            if min(diff, 180 - diff) <= tol:
                keep.append(c)
        return keep

    def _parallelograms(self, level: int) -> list[Domain]:
        corners = self.levels[level]["corners"]
        by_vertex, by_edge = self._adjacency(level)
        out = []
        for v, ts in by_vertex.items():
            if len(ts) != 6:
                continue
            tset = set(ts)
            # edge-neighbours of each triangle within the hexagon (2 shared corners)
            nbrs = {t: {u for u in ts if u != t and len(set(corners[t]) & set(corners[u])) == 2} for t in ts}
            for i, ti in enumerate(ts):
                for tj in ts[i + 1:]:
                    # opposite = three steps apart around the hexagon: not adjacent, and
                    # no common neighbour (two steps apart would yield a trapezoid)
                    if tj in nbrs[ti] or nbrs[ti] & nbrs[tj]:
                        continue
                    extra = []
                    for t in (ti, tj):
                        edge = tuple(sorted(int(x) for x in corners[t] if x != v))
                        nb = [n for n in by_edge[edge] if n not in tset]
                        extra.append(nb[0] if nb else None)
                    if None not in extra:
                        tris = np.array(ts + extra)
                        out.append(Domain(level, tris, self._slices(level, tris), v))
        return out

    def parallelogram_at(self, lat: float, lon: float, level: int,
                         orientation: float | Domain | None = None) -> list[Domain]:
        """Parallelograms whose centre vertex is closest to (lat, lon) [deg]."""
        cands = self.parallelograms(level, orientation=orientation)
        if not cands:
            return []
        centres = np.array([c.center_vertex for c in cands])
        d = (self.vlat[centres] - lat) ** 2 + ((self.vlon[centres] - lon) * np.cos(np.deg2rad(lat))) ** 2
        return [c for c, dd in zip(cands, d) if dd == d.min()]


def select_non_overlapping(domains: list[Domain], seed: list[Domain] | None = None,
                           n_trials: int = 200, rng_seed: int = 0) -> list[Domain]:
    """Select the maximum number of mutually non-overlapping domains (by triangle
    index, so all domains must be of the same level), keeping ``seed``.

    Solved exactly as an integer program (each triangle used at most once). If the
    solver fails, falls back to the best of a min-conflict-first greedy order and
    ``n_trials`` random greedy orders. Seed domains are always kept, first.
    """
    seed = list(seed or [])
    used0 = set().union(*(set(d.triangles.tolist()) for d in seed)) if seed else set()
    free = [d for d in domains if not set(d.triangles.tolist()) & used0]
    if not free:
        return seed
    exact = _max_packing(free)
    if exact is not None:
        return seed + exact
    # min-conflict-first order: candidates overlapping few others go first
    sets = [set(d.triangles.tolist()) for d in free]
    conflicts = [sum(1 for t in sets if s & t) for s in sets]
    orders = [np.argsort(conflicts, kind="stable")]
    rng = np.random.default_rng(rng_seed)
    orders += [rng.permutation(len(free)) for _ in range(n_trials)]
    best: list[Domain] = []
    for order in orders:
        pick = _greedy([free[i] for i in order], used0)
        if len(pick) > len(best):
            best = pick
    return seed + best


def _max_packing(domains: list[Domain], time_limit: float = 60.0) -> list[Domain] | None:
    """Maximum set packing via scipy's MILP solver; None if unavailable/unsuccessful."""
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import csr_matrix
    except ImportError:
        return None
    tri_ids = {t: k for k, t in enumerate(sorted({t for d in domains for t in d.triangles.tolist()}))}
    rows = [tri_ids[t] for d in domains for t in d.triangles.tolist()]
    cols = [j for j, d in enumerate(domains) for _ in d.triangles]
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(tri_ids), len(domains)))
    res = milp(c=-np.ones(len(domains)), constraints=LinearConstraint(A, 0, 1),
               integrality=np.ones(len(domains)), bounds=Bounds(0, 1),
               options={"time_limit": time_limit})
    if res.x is None:
        return None
    return [d for d, x in zip(domains, res.x) if x > 0.5]


def _greedy(domains: list[Domain], used: set[int]) -> list[Domain]:
    used = set(used)
    chosen = []
    for dom in domains:
        tris = set(dom.triangles.tolist())
        if not tris & used:
            chosen.append(dom)
            used |= tris
    return chosen


def to_domains_py(domains: list[Domain], prefix: str = "domain") -> str:
    """Render domains in the format of ``data/domains.py``."""
    lines = ["domains = {"]
    for i, dom in enumerate(domains, 1):
        lines.append(f'    "{prefix}{i:02d}": {{"cells": [')
        lines += [f"        slice({s.start}, {s.stop})," for s in dom.slices]
        lines.append("    ]},")
    lines.append("}")
    return "\n".join(lines)
