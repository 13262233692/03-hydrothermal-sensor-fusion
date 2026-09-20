"""Synthetic sensor-stream generator for hardware-free testing.

Simulates an ROV flying a lawn-mower survey through a Gaussian buoyant
plume and writes one CSV file per "serial port" using the same line
schema as the real sensors, including measurement noise and occasional
spike outliers (so the rejection logic gets exercised).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class PlumeModel:
    """Steady Gaussian plume rising from a vent source."""

    source: Tuple[float, float, float] = (0.0, 0.0, -1500.0)
    rise: float = 60.0          # plume centreline rise above source (m)
    spread_xy: float = 18.0     # horizontal sigma (m)
    spread_z: float = 12.0      # vertical sigma (m)
    peak_h2s: float = 250.0     # umol/kg at the core
    ambient_h2s: float = 0.5
    ambient_temp: float = 2.1   # degC background bottom water
    temp_anomaly: float = 4.5   # degC at the core
    ambient_ph: float = 7.9
    ph_drop: float = 0.9        # pH decrease at the core
    ambient_turb: float = 0.3   # NTU
    turb_peak: float = 12.0     # NTU at the core

    def weight(self, xyz: np.ndarray) -> np.ndarray:
        """Normalised plume intensity 0..1 at positions ``(n, 3)``."""
        src = np.asarray(self.source)
        centre_z = src[2] + self.rise
        dxy = np.sum((xyz[:, :2] - src[:2]) ** 2, axis=1)
        dz = (xyz[:, 2] - centre_z) ** 2
        return np.exp(-0.5 * (dxy / self.spread_xy ** 2
                              + dz / self.spread_z ** 2))

    def sample(self, xyz: np.ndarray) -> Dict[str, np.ndarray]:
        w = self.weight(xyz)
        return {
            "h2s": self.ambient_h2s + self.peak_h2s * w,
            "temperature": self.ambient_temp + self.temp_anomaly * w,
            "ph": self.ambient_ph - self.ph_drop * w,
            "turbidity": self.ambient_turb + self.turb_peak * w,
        }


def lawnmower_trajectory(t: np.ndarray, plume: PlumeModel) -> np.ndarray:
    """Lawn-mower survey path around the vent; returns ``(n, 3)`` xyz."""
    src = np.asarray(plume.source)
    period = 120.0
    phase = (t % period) / period
    leg = np.floor(t / period).astype(int)
    x = src[0] - 60.0 + 120.0 * np.where(phase < 0.5, phase * 2.0,
                                         2.0 - phase * 2.0)
    y = src[1] - 40.0 + 8.0 * (leg % 11)
    z = src[2] + 20.0 + 45.0 * (0.5 + 0.5 * np.sin(2 * np.pi * t / 300.0))
    return np.column_stack([x, y, z])


#: native sampling rates (Hz) and 1-sigma noise per channel
SENSOR_RATES = {"temperature": 4.0, "h2s": 2.0, "ph": 1.0, "turbidity": 1.0}
SENSOR_NOISE = {"temperature": 0.05, "h2s": 2.0, "ph": 0.02, "turbidity": 1.0}
NAV_RATE = 10.0
NAV_NOISE = 0.5  # m


def simulate(output_dir: str, duration: float = 600.0, t0: float = 1.7e9,
             seed: int = 42, spike_fraction: float = 0.01) -> List[str]:
    """Write per-port CSV files plus a nav file; returns the paths."""
    rng = np.random.default_rng(seed)
    plume = PlumeModel()
    os.makedirs(output_dir, exist_ok=True)
    written: List[str] = []

    nav_t = t0 + np.arange(0.0, duration, 1.0 / NAV_RATE)
    nav_xyz = lawnmower_trajectory(nav_t - t0, plume)
    nav_xyz += rng.normal(0.0, NAV_NOISE, nav_xyz.shape)
    nav_path = os.path.join(output_dir, "nav.csv")
    with open(nav_path, "w") as fh:
        fh.write("# channel,epoch_s,x_m,y_m,z_m\n")
        for ti, p in zip(nav_t, nav_xyz):
            fh.write(f"nav,{ti:.3f},{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}\n")
    written.append(nav_path)

    truth_xyz = lawnmower_trajectory(nav_t - t0, plume)
    for channel, rate in SENSOR_RATES.items():
        ts = t0 + np.arange(0.0, duration, 1.0 / rate)
        xyz = np.column_stack([
            np.interp(ts, nav_t, truth_xyz[:, c]) for c in range(3)])
        truth = plume.sample(xyz)[channel]
        meas = truth + rng.normal(0.0, SENSOR_NOISE[channel], ts.shape)
        n_spikes = max(1, int(spike_fraction * ts.size))
        spike_idx = rng.choice(ts.size, size=n_spikes, replace=False)
        scale = {"ph": 1.5, "temperature": 3.0, "h2s": 80.0,
                 "turbidity": 25.0}[channel]
        meas[spike_idx] += rng.choice([-1.0, 1.0], n_spikes) * scale
        path = os.path.join(output_dir, f"{channel}.csv")
        with open(path, "w") as fh:
            fh.write(f"# channel,epoch_s,{channel}\n")
            for ti, v in zip(ts, meas):
                fh.write(f"{channel},{ti:.3f},{v:.5f}\n")
        written.append(path)
    return written
