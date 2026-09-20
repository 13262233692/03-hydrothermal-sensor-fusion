"""Kalman filtering, time alignment and outlier rejection for sensor channels.

Each sensor channel gets an independent constant-velocity Kalman filter, which
naturally handles asynchronous, multi-rate sampling: the filter *predicts* to
whatever timestamp arrives next and *updates* with the measurement.

Outlier rejection uses innovation gating: a measurement whose normalised
innovation squared (NIS = y^2 / S) exceeds a chi-square threshold (1 d.o.f.)
is discarded as a spike -- common on acoustic/serial links and with
electrochemical sensors hitting particles.

Time alignment replays the accepted samples of every channel against a common
time grid, predicting each filter to each grid epoch, yielding a synchronised
multi-channel dataset plus per-channel uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .serial_reader import SensorSample

# Default per-channel noise configuration (tune per instrument).
# process_noise: spectral density of the acceleration driving the value
# (units^2 / s^3) -- larger tracks faster plume gradients while transiting.
# measurement_noise: variance of one reading (units^2).
DEFAULT_SENSOR_CONFIG: Dict[str, Dict[str, float]] = {
    "ph":         {"process_noise": 1e-2, "measurement_noise": 4e-4, "weight": 1.0, "sign": -1.0},
    "h2s":        {"process_noise": 1.0,  "measurement_noise": 4e-2, "weight": 1.0, "sign": +1.0},
    "temperature": {"process_noise": 0.25, "measurement_noise": 2.5e-3, "weight": 1.0, "sign": +1.0},
    "turbidity":  {"process_noise": 0.5,  "measurement_noise": 9e-2, "weight": 0.5, "sign": +1.0},
}


class KalmanFilter1D:
    """Constant-velocity Kalman filter for a single scalar channel.

    State: ``[value, rate]``. The continuous white-noise-acceleration model
    gives the process-noise matrix used in :meth:`predict`.
    """

    def __init__(
        self,
        process_noise: float = 1e-3,
        measurement_noise: float = 1e-2,
        initial_value: float = 0.0,
        initial_variance: float = 1.0,
    ) -> None:
        self.q = float(process_noise)
        self.r = float(measurement_noise)
        self.x = np.array([initial_value, 0.0])
        self.P = np.diag([initial_variance, initial_variance])
        self._H = np.array([[1.0, 0.0]])

    def predict(self, dt: float) -> None:
        if dt <= 0.0:
            return
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = self.q * np.array(
            [[dt ** 3 / 3.0, dt ** 2 / 2.0], [dt ** 2 / 2.0, dt]]
        )
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def innovation(self, z: float) -> Tuple[float, float]:
        """Innovation ``y`` and its variance ``S`` at the current prediction."""
        y = float((z - self._H @ self.x).item())
        S = float((self._H @ self.P @ self._H.T + self.r).item())
        return y, S

    def update(self, z: float) -> None:
        y, S = self.innovation(z)
        K = (self.P @ self._H.T / S).ravel()
        self.x = self.x + K * y
        self.P = (np.eye(2) - np.outer(K, self._H.ravel())) @ self.P

    @property
    def value(self) -> float:
        return float(self.x[0])

    @property
    def variance(self) -> float:
        return float(self.P[0, 0])


@dataclass
class ChannelStats:
    accepted: int = 0
    rejected: int = 0


@dataclass
class FusedSeries:
    """Time-aligned, filtered multi-channel dataset on a common time grid."""

    times: np.ndarray                      # (n_times,)
    channels: Dict[str, np.ndarray]        # name -> (n_times,) filtered value
    variances: Dict[str, np.ndarray]       # name -> (n_times,) filter variance
    anomaly_index: Optional[np.ndarray] = None  # weighted z-score blend

    def channel_names(self) -> List[str]:
        return list(self.channels)


class MultiSensorFusion:
    """Ingest asynchronous samples, gate outliers, align to a time grid."""

    def __init__(
        self,
        sensor_config: Optional[Dict[str, Dict[str, float]]] = None,
        gate_chi2: float = 9.0,
        max_consecutive_rejects: int = 5,
    ) -> None:
        self.config = dict(DEFAULT_SENSOR_CONFIG)
        if sensor_config:
            for name, cfg in sensor_config.items():
                self.config.setdefault(name, {}).update(cfg)
        self.gate_chi2 = float(gate_chi2)  # 9.0 ~= 3-sigma for 1 d.o.f.
        # Lock-out recovery: after this many consecutive rejects the gate is
        # lifted once, so a genuine regime change cannot starve the filter.
        self.max_consecutive_rejects = int(max_consecutive_rejects)
        self._consecutive_rejects: Dict[str, int] = {}
        self._filters: Dict[str, KalmanFilter1D] = {}
        self._last_time: Dict[str, float] = {}
        self._accepted: Dict[str, List[SensorSample]] = {}
        self.stats: Dict[str, ChannelStats] = {}

    # ------------------------------------------------------------------ ingest
    def _make_filter(self, sensor: str, first_value: float) -> KalmanFilter1D:
        cfg = self.config.get(sensor, {})
        return KalmanFilter1D(
            process_noise=cfg.get("process_noise", 1e-3),
            measurement_noise=cfg.get("measurement_noise", 1e-2),
            initial_value=first_value,
        )

    def ingest(self, sample: SensorSample) -> bool:
        """Process one sample online. Returns True if accepted (not a spike)."""
        name = sample.sensor
        self.stats.setdefault(name, ChannelStats())
        self._accepted.setdefault(name, [])

        if name not in self._filters:
            self._filters[name] = self._make_filter(name, sample.value)
            self._last_time[name] = sample.timestamp
            self._accepted[name].append(sample)
            self.stats[name].accepted += 1
            return True

        dt = sample.timestamp - self._last_time[name]
        if dt < 0.0:
            return False  # out-of-order packet; ignore

        kf = self._filters[name]
        kf.predict(dt)
        self._last_time[name] = sample.timestamp

        y, S = kf.innovation(sample.value)
        streak = self._consecutive_rejects.get(name, 0)
        if y * y / S > self.gate_chi2 and streak < self.max_consecutive_rejects:
            self._consecutive_rejects[name] = streak + 1
            self.stats[name].rejected += 1
            return False  # spike rejected; state stays at the prediction

        self._consecutive_rejects[name] = 0
        kf.update(sample.value)
        self._accepted[name].append(sample)
        self.stats[name].accepted += 1
        return True

    def ingest_all(self, samples: Iterable[SensorSample]) -> None:
        for sample in samples:
            self.ingest(sample)

    # ----------------------------------------------------------------- align
    def time_span(self) -> Tuple[float, float]:
        lo = min(s.timestamp for ch in self._accepted.values() for s in ch)
        hi = max(s.timestamp for ch in self._accepted.values() for s in ch)
        return lo, hi

    def align(self, grid_times: np.ndarray) -> FusedSeries:
        """Replay accepted samples and snapshot each channel at grid epochs."""
        grid_times = np.asarray(grid_times, dtype=float)
        channels: Dict[str, np.ndarray] = {}
        variances: Dict[str, np.ndarray] = {}

        for name, samples in self._accepted.items():
            samples = sorted(samples, key=lambda s: s.timestamp)
            kf = self._make_filter(name, samples[0].value)
            values = np.full(grid_times.shape, np.nan)
            var = np.full(grid_times.shape, np.nan)
            idx = 0
            last_t = samples[0].timestamp
            for gi, t in enumerate(grid_times):
                while idx < len(samples) and samples[idx].timestamp <= t:
                    dt = samples[idx].timestamp - last_t
                    kf.predict(max(dt, 0.0))
                    kf.update(samples[idx].value)
                    last_t = samples[idx].timestamp
                    idx += 1
                kf.predict(max(t - last_t, 0.0))
                last_t = t
                values[gi] = kf.value
                var[gi] = kf.variance
            channels[name] = values
            variances[name] = var

        return FusedSeries(
            times=grid_times,
            channels=channels,
            variances=variances,
            anomaly_index=self._anomaly_index(channels),
        )

    # ------------------------------------------------------------------ fuse
    def _anomaly_index(self, channels: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """Weighted z-score blend across channels (hydrothermal anomaly index)."""
        if not channels:
            return None
        acc, wsum = None, 0.0
        for name, values in channels.items():
            cfg = self.config.get(name, {})
            weight = cfg.get("weight", 1.0)
            sign = cfg.get("sign", 1.0)
            mu, sigma = np.nanmean(values), np.nanstd(values)
            if not np.isfinite(sigma) or sigma == 0.0:
                continue
            z = sign * (values - mu) / sigma
            acc = weight * z if acc is None else acc + weight * z
            wsum += weight
        return acc / wsum if wsum > 0 else None
