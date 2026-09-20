"""3-D plume reconstruction from fused, georeferenced sensor data.

The fused time series (already time-aligned and outlier-free) is combined
with the ROV navigation solution into scattered ``(x, y, z, value)``
samples, then interpolated onto a regular 3-D grid with ordinary Kriging.

Kriging is solved in a local neighbourhood (k nearest samples via a
KD-tree) which keeps the linear systems small and the method usable for
tens of thousands of scattered samples.  An exponential variogram model
is used by default; the sill defaults to the sample variance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)


def exponential_variogram(h: np.ndarray, range_: float, sill: float,
                          nugget: float = 0.0) -> np.ndarray:
    """Exponential variogram ``nugget + sill * (1 - exp(-3h/range))``."""
    h = np.asarray(h, dtype=float)
    return nugget + sill * (1.0 - np.exp(-3.0 * h / max(range_, 1e-12)))


@dataclass
class ProfileGrid:
    """A regular 3-D grid holding one interpolated channel."""

    name: str
    x: np.ndarray          # (nx,)
    y: np.ndarray          # (ny,)
    z: np.ndarray          # (nz,)
    values: np.ndarray     # (nz, ny, nx) interpolated estimates
    variance: np.ndarray   # (nz, ny, nx) kriging variance
    n_samples: int = 0


class OrdinaryKriging3D:
    """Local-neighbourhood ordinary Kriging in 3-D."""

    def __init__(self, range_: float, sill: Optional[float] = None,
                 nugget: float = 0.0, max_neighbors: int = 24):
        if range_ <= 0:
            raise ValueError("variogram range must be positive")
        self.range_ = float(range_)
        self.sill = sill
        self.nugget = float(nugget)
        self.max_neighbors = int(max_neighbors)

    def _krige_block(self, tree_pts: np.ndarray, values: np.ndarray,
                     targets: np.ndarray, sill: float) -> Tuple[np.ndarray,
                                                                np.ndarray]:
        k = min(self.max_neighbors, tree_pts.shape[0])
        tree = cKDTree(tree_pts)
        dists, idx = tree.query(targets, k=k)
        est = np.empty(targets.shape[0])
        var = np.empty(targets.shape[0])
        for row in range(targets.shape[0]):
            nb = np.atleast_1d(idx[row])
            pts = tree_pts[nb]
            vals = values[nb]
            d = np.linalg.norm(pts[None, :, :] - pts[:, None, :], axis=2)
            C = sill - exponential_variogram(d, self.range_, sill,
                                             self.nugget)
            A = np.empty((k + 1, k + 1))
            A[:k, :k] = C
            A[:k, k] = 1.0
            A[k, :k] = 1.0
            A[k, k] = 0.0
            d0 = np.linalg.norm(pts - targets[row], axis=1)
            c = sill - exponential_variogram(d0, self.range_, sill,
                                             self.nugget)
            rhs = np.concatenate([c, [1.0]])
            try:
                weights = np.linalg.solve(A, rhs)
            except np.linalg.LinAlgError:
                weights = np.linalg.lstsq(A, rhs, rcond=None)[0]
            est[row] = weights[:k] @ vals
            var[row] = max(sill - weights[:k] @ c - weights[k], 0.0)
        return est, var

    def predict(self, points: np.ndarray, values: np.ndarray,
                targets: np.ndarray,
                block_size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate ``values`` at scattered ``points`` onto ``targets``."""
        points = np.asarray(points, dtype=float)
        values = np.asarray(values, dtype=float)
        sill = float(self.sill) if self.sill is not None else \
            max(float(np.var(values)), 1e-12)
        est = np.empty(targets.shape[0])
        var = np.empty(targets.shape[0])
        for start in range(0, targets.shape[0], block_size):
            stop = min(start + block_size, targets.shape[0])
            est[start:stop], var[start:stop] = self._krige_block(
                points, values, targets[start:stop], sill)
        return est, var


def build_profile(positions: np.ndarray, values: np.ndarray, name: str,
                  grid_shape: Tuple[int, int, int] = (24, 24, 12),
                  margin: float = 5.0,
                  variogram_range: Optional[float] = None,
                  max_neighbors: int = 24) -> ProfileGrid:
    """Grid one fused channel into a 3-D concentration profile.

    ``positions`` is ``(n, 3)`` ROV x/y/z, ``values`` the fused channel
    (NaN rows are dropped).  ``grid_shape`` is ``(nx, ny, nz)``.
    """
    positions = np.asarray(positions, dtype=float)
    values = np.asarray(values, dtype=float).ravel()
    good = np.isfinite(values) & np.all(np.isfinite(positions), axis=1)
    pts = positions[good]
    vals = values[good]
    if pts.shape[0] < 8:
        raise ValueError(f"channel {name!r}: only {pts.shape[0]} usable "
                         "georeferenced samples, need >= 8")

    lo = pts.min(axis=0) - margin
    hi = pts.max(axis=0) + margin
    nx, ny, nz = grid_shape
    xs = np.linspace(lo[0], hi[0], nx)
    ys = np.linspace(lo[1], hi[1], ny)
    zs = np.linspace(lo[2], hi[2], nz)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    targets = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    if variogram_range is None:
        extent = np.linalg.norm(hi - lo)
        variogram_range = max(extent / 4.0, 1.0)
    kriging = OrdinaryKriging3D(range_=variogram_range,
                                max_neighbors=max_neighbors)
    log.info("kriging %-11s n=%5d grid=%s range=%.1f m",
             name, pts.shape[0], grid_shape, variogram_range)
    est, var = kriging.predict(pts, vals, targets)

    return ProfileGrid(name=name, x=xs, y=ys, z=zs,
                       values=est.reshape(nx, ny, nz).transpose(2, 1, 0),
                       variance=var.reshape(nx, ny, nz).transpose(2, 1, 0),
                       n_samples=int(pts.shape[0]))


def build_all_profiles(fusion_result, channels: Optional[list] = None,
                       grid_shape: Tuple[int, int, int] = (24, 24, 12),
                       margin: float = 5.0) -> Dict[str, ProfileGrid]:
    """Build 3-D grids for every fused channel that has nav coverage."""
    if fusion_result.nav is None:
        raise ValueError("fusion result carries no navigation data")
    names = channels or sorted(fusion_result.channels)
    grids: Dict[str, ProfileGrid] = {}
    for name in names:
        channel = fusion_result.channels.get(name)
        if channel is None:
            log.warning("channel %r missing from fusion result, skipped",
                        name)
            continue
        try:
            grids[name] = build_profile(fusion_result.nav, channel.values,
                                        name, grid_shape=grid_shape,
                                        margin=margin)
        except ValueError as exc:
            log.warning("%s", exc)
    if not grids:
        raise ValueError("no profiles could be built")
    return grids
