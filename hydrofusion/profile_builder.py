"""Build 3-D concentration profiles from fused, geo-referenced sensor series.

Positions of the ROV at the common time-grid epochs are obtained by linearly
interpolating the nav fixes carried on the incoming samples. Each channel is
then interpolated onto a regular 3-D grid with Ordinary Kriging (exponential
variogram, local neighbourhoods via KD-tree). When a channel has too few
samples for Kriging, it falls back to nearest-neighbour assignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .kalman_fusion import FusedSeries
from .serial_reader import SensorSample


class OrdinaryKriging3D:
    """Ordinary Kriging in 3-D with an exponential variogram.

    Semivariogram: ``gamma(h) = nugget + (sill - nugget) * (1 - exp(-3h/range))``.
    Predictions are made with the ``max_neighbors`` nearest samples to keep
    the kriging systems small and the interpolation local.
    """

    def __init__(
        self,
        range: float = 8.0,
        sill: float = 1.0,
        nugget: float = 1e-2,
        max_neighbors: int = 32,
    ) -> None:
        self.range = float(range)
        self.sill = float(sill)
        self.nugget = float(nugget)
        self.max_neighbors = int(max_neighbors)

    def _covariance(self, d: np.ndarray) -> np.ndarray:
        return (self.sill - self.nugget) * np.exp(-3.0 * d / self.range)

    def predict(
        self,
        points: np.ndarray,
        values: np.ndarray,
        targets: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Kriging estimate and variance at ``targets`` (n,3) arrays."""
        points = np.asarray(points, dtype=float)
        values = np.asarray(values, dtype=float)
        targets = np.asarray(targets, dtype=float)
        tree = cKDTree(points)
        k = min(self.max_neighbors, len(points))
        estimates = np.empty(len(targets))
        variances = np.empty(len(targets))

        for i, target in enumerate(targets):
            dist, idx = tree.query(target, k=k)
            dist = np.atleast_1d(dist)
            idx = np.atleast_1d(idx)
            local = points[idx]
            D = np.linalg.norm(local[None, :, :] - local[:, None, :], axis=-1)
            C = self._covariance(D) + self.nugget * np.eye(len(idx))
            c = self._covariance(dist)
            # Ordinary kriging system with Lagrange multiplier.
            A = np.zeros((k + 1, k + 1))
            A[:k, :k] = C
            A[:k, k] = 1.0
            A[k, :k] = 1.0
            b = np.concatenate([c, [1.0]])
            try:
                weights = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                weights = np.linalg.lstsq(A, b, rcond=None)[0]
            w = weights[:k]
            estimates[i] = w @ values[idx]
            variances[i] = max(
                self.sill - w @ c - weights[k], 0.0
            )
        return estimates, variances


@dataclass
class ProfileResult:
    """Gridded 3-D fields plus coordinate vectors."""

    x: np.ndarray                       # (nx,)
    y: np.ndarray                       # (ny,)
    z: np.ndarray                       # (nz,)
    fields: Dict[str, np.ndarray]       # name -> (nz, ny, nx)
    variances: Dict[str, np.ndarray] = field(default_factory=dict)


class ProfileBuilder:
    """Interpolate fused channel time series onto a regular 3-D grid."""

    def __init__(
        self,
        grid_shape: Tuple[int, int, int] = (24, 24, 16),
        kriging: Optional[OrdinaryKriging3D] = None,
        min_samples_for_kriging: int = 8,
    ) -> None:
        self.grid_shape = tuple(int(n) for n in grid_shape)
        self.kriging = kriging or OrdinaryKriging3D()
        self.min_samples_for_kriging = int(min_samples_for_kriging)

    # ------------------------------------------------------------- positions
    @staticmethod
    def positions_at(
        samples: Sequence[SensorSample], times: np.ndarray
    ) -> np.ndarray:
        """Interpolate ROV xyz from nav-bearing samples to ``times``."""
        nav = [s for s in samples if s.has_position]
        if not nav:
            raise ValueError("no samples carry ROV position (x, y, z)")
        nav.sort(key=lambda s: s.timestamp)
        t = np.array([s.timestamp for s in nav])
        xyz = np.empty((len(times), 3))
        for axis, attr in enumerate(("x", "y", "z")):
            v = np.array([getattr(s, attr) for s in nav])
            xyz[:, axis] = np.interp(times, t, v)
        return xyz

    # ------------------------------------------------------------------ grid
    def _grid(
        self, points: np.ndarray, bounds: Optional[Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pad = 0.05
        lo = points.min(axis=0) if bounds is None else np.asarray(bounds[0])
        hi = points.max(axis=0) if bounds is None else np.asarray(bounds[1])
        span = np.maximum(hi - lo, 1e-6)
        lo, hi = lo - pad * span, hi + pad * span
        nx, ny, nz = self.grid_shape
        xs = np.linspace(lo[0], hi[0], nx)
        ys = np.linspace(lo[1], hi[1], ny)
        zs = np.linspace(lo[2], hi[2], nz)
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        targets = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
        return xs, ys, zs, targets

    # ----------------------------------------------------------------- build
    def build(
        self,
        fused: FusedSeries,
        samples: Sequence[SensorSample],
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> ProfileResult:
        points = self.positions_at(samples, fused.times)
        xs, ys, zs, targets = self._grid(points, bounds)
        nx, ny, nz = self.grid_shape
        shape = (nz, ny, nx)  # NetCDF axis order (z, y, x)

        fields: Dict[str, np.ndarray] = {}
        variances: Dict[str, np.ndarray] = {}
        for name, series in fused.channels.items():
            mask = np.isfinite(series)
            if mask.sum() < 3:
                continue
            pts, vals = points[mask], series[mask]
            if mask.sum() >= self.min_samples_for_kriging:
                est, var = self.kriging.predict(pts, vals, targets)
            else:  # too few samples: nearest neighbour
                _, idx = cKDTree(pts).query(targets)
                est = vals[idx]
                var = np.full(len(targets), np.nan)
            # targets were ravelled from an (nx, ny, nz) ij-grid; reorder to
            # the (nz, ny, nx) storage layout.
            fields[name] = est.reshape(nx, ny, nz).transpose(2, 1, 0)
            variances[name] = var.reshape(nx, ny, nz).transpose(2, 1, 0)

        return ProfileResult(x=xs, y=ys, z=zs, fields=fields, variances=variances)
