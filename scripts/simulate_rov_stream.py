#!/usr/bin/env python3
"""Generate a synthetic ROV sensor stream (CSV) for testing the pipeline.

The ROV flies a lawn-mower survey through a synthetic 3-D Gaussian plume.
Four sensors sample at different rates with realistic noise and occasional
spikes, each line stamped with the ROV position.
"""

from __future__ import annotations

import argparse

import numpy as np

SENSOR_RATES = {"temperature": 5.0, "turbidity": 4.0, "h2s": 2.0, "ph": 1.0}
SENSOR_NOISE = {"temperature": 0.05, "turbidity": 0.3, "h2s": 0.2, "ph": 0.02}


def rov_position(t: float) -> np.ndarray:
    """Lawn-mower survey path, ~40 x 30 m, gently varying depth."""
    x = 20.0 * np.sin(2 * np.pi * t / 120.0)
    y = 15.0 * np.sin(2 * np.pi * t / 45.0)
    z = -100.0 + 8.0 * np.sin(2 * np.pi * t / 90.0)
    return np.array([x, y, z])


def plume_field(pos: np.ndarray) -> float:
    """Normalised plume intensity: Gaussian blob near (0, 0, -95)."""
    center = np.array([0.0, 0.0, -95.0])
    sigma = np.array([8.0, 8.0, 5.0])
    d2 = np.sum(((pos - center) / sigma) ** 2)
    return np.exp(-0.5 * d2)


def true_value(sensor: str, intensity: float) -> float:
    return {
        "temperature": 2.0 + 25.0 * intensity,
        "turbidity": 0.5 + 12.0 * intensity,
        "h2s": 0.1 + 40.0 * intensity,
        "ph": 8.0 - 3.0 * intensity,
    }[sensor]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/simulated_stream.csv")
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument("--spike-prob", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for sensor, rate in SENSOR_RATES.items():
        for t in np.arange(0.0, args.duration, 1.0 / rate):
            pos = rov_position(t)
            intensity = plume_field(pos)
            value = true_value(sensor, intensity)
            value += rng.normal(0.0, SENSOR_NOISE[sensor])
            if rng.random() < args.spike_prob:  # telemetry spike
                value += rng.normal(0.0, 10.0 * SENSOR_NOISE[sensor] + 5.0)
            rows.append((t, sensor, value, *pos))

    rows.sort(key=lambda r: r[0])
    with open(args.output, "w", encoding="ascii") as fh:
        fh.write("# timestamp,sensor,value,x,y,z\n")
        for t, sensor, value, x, y, z in rows:
            fh.write(f"{t:.3f},{sensor},{value:.6f},{x:.3f},{y:.3f},{z:.3f}\n")
    print(f"wrote {len(rows)} samples to {args.output}")


if __name__ == "__main__":
    main()
