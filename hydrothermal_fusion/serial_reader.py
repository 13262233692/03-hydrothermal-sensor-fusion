"""Serial CSV stream acquisition for the ROV hydrothermal sensor suite.

Each sensor (pH, H2S, temperature, turbidity) and the ROV navigation
system emit CSV lines over RS-232 at their own native sampling rate.
Every line follows the schema::

    <channel>,<epoch_seconds>,<v1>[,<v2>,...]

Examples::

    h2s,1712345678.125,143.2
    nav,1712345678.050,1201.4,502.9,-1480.2

Lines starting with ``#`` are treated as comments.  Malformed lines are
skipped and counted so a noisy serial link cannot crash the pipeline.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Dict, IO, Iterator, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

#: Science channels carried by the ROV payload.
CHANNELS: Tuple[str, ...] = ("ph", "h2s", "temperature", "turbidity")

#: Navigation channel (ROV x/y/z position, metres, local frame).
NAV_CHANNEL = "nav"


@dataclass
class SensorSample:
    """One parsed CSV record from the stream."""

    channel: str
    timestamp: float
    values: Tuple[float, ...]
    line_no: int = 0


def parse_csv_line(line: str, line_no: int = 0) -> Optional[SensorSample]:
    """Parse one CSV line into a :class:`SensorSample`.

    Returns ``None`` for blank lines and comments, raises ``ValueError``
    for malformed records.
    """
    text = line.strip()
    if not text or text.startswith("#"):
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) < 3:
        raise ValueError(f"line {line_no}: expected >=3 fields, got {len(parts)}")
    channel = parts[0].lower()
    timestamp = float(parts[1])
    values = tuple(float(v) for v in parts[2:] if v != "")
    if not values:
        raise ValueError(f"line {line_no}: no numeric payload")
    return SensorSample(channel=channel, timestamp=timestamp,
                        values=values, line_no=line_no)


class CSVStreamReader:
    """Iterate :class:`SensorSample` records from any text stream.

    Works with a pyserial ``TextIOWrapper`` around a serial port just as
    well as with a plain file object, which makes replay/testing trivial.
    """

    def __init__(self, stream: IO[str], source: str = "stream"):
        self._stream = stream
        self.source = source
        self.bad_lines = 0

    def samples(self) -> Iterator[SensorSample]:
        for line_no, line in enumerate(self._stream, start=1):
            try:
                sample = parse_csv_line(line, line_no)
            except ValueError as exc:
                self.bad_lines += 1
                log.warning("skipping malformed line from %s: %s",
                            self.source, exc)
                continue
            if sample is not None:
                yield sample


class SerialPortReader(threading.Thread):
    """Background thread reading one serial port into a shared queue.

    Sensors on separate ports (or a single muxed port) can each run in
    their own thread; samples are funnelled into ``out_queue`` as
    :class:`SensorSample` objects.
    """

    def __init__(self, port: str, out_queue: "queue.Queue[SensorSample]",
                 baudrate: int = 115200, timeout: float = 1.0):
        super().__init__(daemon=True, name=f"serial-{port}")
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.out_queue = out_queue
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:  # pragma: no cover - requires hardware
        import serial  # pyserial, imported lazily for offline replay

        with serial.Serial(self.port, self.baudrate,
                           timeout=self.timeout) as ser:
            import io
            text_stream = io.TextIOWrapper(ser, encoding="ascii",
                                           errors="replace")
            reader = CSVStreamReader(text_stream, source=self.port)
            for sample in reader.samples():
                if self._stop_event.is_set():
                    break
                self.out_queue.put(sample)


class MultiChannelCollector:
    """Accumulate samples from any number of readers, grouped by channel."""

    def __init__(self) -> None:
        self._data: Dict[str, List[SensorSample]] = {}

    def add(self, sample: SensorSample) -> None:
        self._data.setdefault(sample.channel, []).append(sample)

    def add_from(self, samples: Iterator[SensorSample]) -> int:
        count = 0
        for sample in samples:
            self.add(sample)
            count += 1
        return count

    @property
    def channels(self) -> List[str]:
        return sorted(self._data)

    def to_arrays(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Return ``{channel: (timestamps, values)}`` sorted by time.

        Single-value channels yield a 1-D value array; the nav channel
        yields an ``(n, 3)`` array.
        """
        out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for channel, samples in self._data.items():
            samples = sorted(samples, key=lambda s: s.timestamp)
            ts = np.array([s.timestamp for s in samples], dtype=float)
            vals = np.array([s.values[0] if len(s.values) == 1
                             else s.values for s in samples], dtype=float)
            out[channel] = (ts, vals)
        return out


def read_csv_file(path: str) -> MultiChannelCollector:
    """Convenience helper: load one CSV file (possibly multi-channel)."""
    collector = MultiChannelCollector()
    with open(path, "r", encoding="ascii", errors="replace") as fh:
        reader = CSVStreamReader(fh, source=path)
        collector.add_from(reader.samples())
        if reader.bad_lines:
            log.info("%s: %d malformed lines skipped", path, reader.bad_lines)
    return collector
