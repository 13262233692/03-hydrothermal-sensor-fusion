"""Serial CSV stream readers for ROV sensor telemetry.

CSV line format (one sample per line)::

    timestamp,sensor,value[,x,y,z]

- ``timestamp``: UNIX epoch seconds (float) or ISO-8601 string.
- ``sensor``:    channel name, e.g. ``ph``, ``h2s``, ``temperature``, ``turbidity``.
- ``value``:     measured scalar (native units).
- ``x, y, z``:   optional ROV position in metres (z positive up, depth = -z).

Two reader implementations share one interface:

- :class:`SerialCSVReader` -- live stream from a pySerial port.
- :class:`FileCSVReader`   -- replay a recorded CSV file (testing / post-processing).
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, Optional, TextIO

import numpy as np


@dataclass
class SensorSample:
    """One parsed sensor reading."""

    timestamp: float
    sensor: str
    value: float
    x: float = np.nan
    y: float = np.nan
    z: float = np.nan

    @property
    def has_position(self) -> bool:
        return bool(np.isfinite([self.x, self.y, self.z]).all())


def parse_timestamp(raw: str) -> float:
    """Accept epoch seconds or ISO-8601 and return epoch seconds."""
    raw = raw.strip()
    try:
        return float(raw)
    except ValueError:
        pass
    text = raw.replace("Z", "+00:00")
    return datetime.fromisoformat(text).timestamp()


def parse_csv_line(line: str) -> Optional[SensorSample]:
    """Parse one CSV line; return ``None`` for blanks/comments/bad rows."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        return None
    try:
        timestamp = parse_timestamp(parts[0])
        sensor = parts[1].lower()
        value = float(parts[2])
        xyz = [float(p) for p in parts[3:6]] if len(parts) >= 6 else []
    except (ValueError, TypeError):
        return None
    if not np.isfinite(value):
        return None
    if xyz:
        return SensorSample(timestamp, sensor, value, *xyz)
    return SensorSample(timestamp, sensor, value)


class _StreamReaderBase:
    """Shared line-parsing loop over a text stream."""

    def __init__(self) -> None:
        self.bad_lines = 0

    def _iter_stream(self, stream: TextIO) -> Iterator[SensorSample]:
        for line in stream:
            sample = parse_csv_line(line)
            if sample is None:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    self.bad_lines += 1
                continue
            yield sample

    def samples(self) -> Iterator[SensorSample]:
        raise NotImplementedError


class SerialCSVReader(_StreamReaderBase):
    """Read a live CSV stream from a serial port (requires pySerial)."""

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        timeout: float = 1.0,
        encoding: str = "ascii",
    ) -> None:
        super().__init__()
        import serial  # deferred so replay works without pySerial installed

        self._serial = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        self._stream = io.TextIOWrapper(
            io.BufferedRWPair(self._serial, self._serial),
            encoding=encoding,
            errors="replace",
            newline="\n",
        )

    def samples(self, duration: Optional[float] = None) -> Iterator[SensorSample]:
        """Yield samples; stop after ``duration`` seconds if given."""
        deadline = None if duration is None else time.monotonic() + duration
        for sample in self._iter_stream(self._stream):
            if deadline is not None and time.monotonic() > deadline:
                return
            yield sample

    def close(self) -> None:
        self._serial.close()

    def __enter__(self) -> "SerialCSVReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class FileCSVReader(_StreamReaderBase):
    """Replay a recorded CSV file with the same interface as the serial reader.

    Set ``realtime=True`` to pace replay by the timestamps in the file
    (scaled by ``speed``), which is handy for exercising streaming code paths.
    """

    def __init__(self, path: str, realtime: bool = False, speed: float = 1.0) -> None:
        super().__init__()
        self.path = path
        self.realtime = realtime
        self.speed = speed

    def samples(self, duration: Optional[float] = None) -> Iterator[SensorSample]:
        del duration  # file length bounds the replay
        t0_file: Optional[float] = None
        t0_wall: Optional[float] = None
        with open(self.path, "r", encoding="utf-8") as fh:
            for sample in self._iter_stream(fh):
                if self.realtime:
                    if t0_file is None:
                        t0_file = sample.timestamp
                        t0_wall = time.monotonic()
                    else:
                        target = t0_wall + (sample.timestamp - t0_file) / self.speed
                        delay = target - time.monotonic()
                        if delay > 0:
                            time.sleep(delay)
                yield sample
