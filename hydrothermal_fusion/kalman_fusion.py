"""Time alignment, outlier rejection and Kalman fusion.

The four science sensors sample asynchronously (e.g. temperature at 4 Hz,
H2S at 2 Hz, pH and turbidity at 1 Hz).  This module

1. pre-filters each channel with a rolling-median (Hampel) screen,
2. runs a constant-velocity Kalman filter per channel with chi-square
   innovation gating (residual outliers are rejected, never fused),
3. evaluates the filtered state on a common, uniformly spaced time grid
   so all channels are time-aligned for the profile builder.

Grid points farther than ``max_gap`` seconds from any real observation
are marked NaN instead of being extrapolated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class ChannelConfig:
    """Per-channel filter tuning."""

    name: str
    process_noise: float = 1e-3      # q, spectral density of the accel. noise
    measurement_noise: float = 1e-2  # r, sensor variance
    gate_chi2: float = 9.0           # innovation gate (~3 sigma, 1 dof)
    hampel_window: int = 7           # rolling-median window (samples)
    hampel_sigmas: float = 4.0       # Hampel threshold in MAD sigmas


# Sensible defaults for the ROV payload (tuned for the simulator and
# typical off-the-shelf deep-sea sensors).
DEFAULT_CONFIGS: Dict[str, ChannelConfig] = {
    "ph":         ChannelConfig("ph",         process_noise=1e-4,
                                measurement_noise=4e-4),
    "h2s":        ChannelConfig("h2s",        process_noise=1.0,
                                measurement_noise=4.0),
    "temperature": ChannelConfig("temperature", process_noise=1e-3,
                                 measurement_noise=2.5e-3),
    "turbidity":  ChannelConfig("turbidity",  process_noise=1e-2,
                                measurement_noise=1.0),
}


def hampel_mask(values: np.ndarray, window: int = 7,
                n_sigmas: float = 4.0) -> np.ndarray:
    """Boolean mask of inliers from a Hampel (rolling median + MAD) filter."""
    n = values.size
    if n < 3 or window < 3:
        return np.ones(n, dtype=bool)
    half = window // 2
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window_vals = values[lo:hi]
        med = np.median(window_vals)
        mad = np.median(np.abs(window_vals - med))
        sigma = 1.4826 * mad
        deviation = abs(values[i] - med)
        if sigma > 0:
            if deviation > n_sigmas * sigma:
                mask[i] = False
        elif deviation > 0:
            # zero MAD: half the window equals the median exactly,
            # so any deviation at all is an outlier
            mask[i] = False
    return mask


class KalmanFilter1D:
    """Constant-velocity (local linear trend) Kalman filter, one channel.

    State ``x = [value, rate]``; the transition matrix is rebuilt for each
    (variable) ``dt`` so asynchronous samples are handled natively.
    """

    def __init__(self, process_noise: float, measurement_noise: float,
                 gate_chi2: float = 9.0):
        self.q = process_noise
        self.r = measurement_noise
        self.gate_chi2 = gate_chi2
        self.x: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.rejected = 0
        self.accepted = 0

    def _predict(self, dt: float) -> None:
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = self.q * np.array([[dt ** 3 / 3.0, dt ** 2 / 2.0],
                               [dt ** 2 / 2.0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def filter(self, times: np.ndarray,
               values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run the filter; returns filtered (values, std) at ``times``."""
        n = times.size
        out = np.full(n, np.nan)
        out_std = np.full(n, np.nan)
        self.x = np.array([values[0], 0.0])
        self.P = np.diag([self.r * 10.0, 1.0])
        last_t = times[0]
        for i in range(n):
            dt = max(times[i] - last_t, 0.0)
            last_t = times[i]
            if dt > 0.0:
                self._predict(dt)
            innovation = values[i] - self.x[0]
            S = self.P[0, 0] + self.r
            nis = innovation ** 2 / S
            if nis <= self.gate_chi2:
                K = self.P[:, 0] / S
                self.x = self.x + K * innovation
                self.P = self.P - np.outer(K, self.P[0, :])
                self.accepted += 1
            else:
                self.rejected += 1
            out[i] = self.x[0]
            out_std[i] = np.sqrt(max(self.P[0, 0], 0.0))
        return out, out_std


@dataclass
class FusedChannel:
    """One channel resampled onto the common time grid."""

    name: str
    values: np.ndarray          # filtered values on the common grid
    std: np.ndarray             # 1-sigma estimate uncertainty
    n_input: int = 0
    n_rejected_hampel: int = 0
    n_rejected_gate: int = 0


@dataclass
class FusionResult:
    """Time-aligned, filtered multi-channel product."""

    time: np.ndarray                       # common time grid (epoch s)
    channels: Dict[str, FusedChannel] = field(default_factory=dict)
    nav: Optional[np.ndarray] = None       # (n, 3) x/y/z on the grid


class KalmanFusion:
    """Fuse asynchronous sensor channels onto a common time base."""

    def __init__(self, configs: Optional[Dict[str, ChannelConfig]] = None,
                 output_rate_hz: float = 1.0, max_gap: float = 5.0):
        self.configs = dict(DEFAULT_CONFIGS)
        if configs:
            self.configs.update(configs)
        self.output_rate_hz = output_rate_hz
        self.max_gap = max_gap

    def fuse(self,
             channel_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
             nav_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
             ) -> FusionResult:
        science = {k: v for k, v in channel_data.items()
                   if k in self.configs and v[0].size >= 3}
        if not science:
            raise ValueError("no usable science channels supplied")

        t_min = min(t[0][0] for t in science.values())
        t_max = max(t[0][-1] for t in science.values())
        dt = 1.0 / self.output_rate_hz
        grid = np.arange(t_min, t_max + 0.5 * dt, dt)
        result = FusionResult(time=grid)

        for name, (times, values) in sorted(science.items()):
            cfg = self.configs[name]
            values = np.asarray(values, dtype=float).ravel()
            keep = hampel_mask(values, cfg.hampel_window, cfg.hampel_sigmas)
            n_hampel = int((~keep).sum())
            kf = KalmanFilter1D(cfg.process_noise, cfg.measurement_noise,
                                cfg.gate_chi2)
            filt, std = kf.filter(times[keep], values[keep])
            aligned, aligned_std = self._resample(times[keep], filt, std, grid)
            result.channels[name] = FusedChannel(
                name=name, values=aligned, std=aligned_std,
                n_input=int(times.size), n_rejected_hampel=n_hampel,
                n_rejected_gate=kf.rejected)
            log.info("channel %-11s n=%5d hampel=%3d gated=%3d",
                     name, times.size, n_hampel, kf.rejected)

        if nav_data is not None:
            result.nav = self._align_nav(nav_data, grid)
        return result

    def _resample(self, obs_t: np.ndarray, filt: np.ndarray,
                  std: np.ndarray,
                  grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate filtered estimates onto the grid, honouring max_gap."""
        values = np.interp(grid, obs_t, filt)
        sigma = np.interp(grid, obs_t, std)
        nearest = obs_t[np.searchsorted(obs_t, grid).clip(1, obs_t.size - 1)]
        prev = obs_t[(np.searchsorted(obs_t, grid) - 1).clip(0)]
        gap = np.minimum(np.abs(grid - nearest), np.abs(grid - prev))
        gap[grid < obs_t[0]] = np.inf
        gap[grid > obs_t[-1]] = np.inf
        bad = gap > self.max_gap
        values[bad] = np.nan
        sigma[bad] = np.nan
        return values, sigma

    @staticmethod
    def _align_nav(nav_data: Tuple[np.ndarray, np.ndarray],
                   grid: np.ndarray) -> np.ndarray:
        times, xyz = nav_data
        xyz = np.atleast_2d(np.asarray(xyz, dtype=float))
        out = np.full((grid.size, 3), np.nan)
        for col in range(3):
            out[:, col] = np.interp(grid, times, xyz[:, col],
                                    left=np.nan, right=np.nan)
        return out
