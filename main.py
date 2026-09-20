#!/usr/bin/env python3
"""End-to-end pipeline: serial CSV -> Kalman fusion -> 3-D kriging -> NetCDF.

Examples
--------
Replay a recorded stream and build the profile::

    python main.py --input data/simulated_stream.csv --output plume.nc

Read a live serial port for 120 s::

    python main.py --port /dev/ttyUSB0 --baud 115200 --duration 120 --output plume.nc
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from hydrofusion import (
    FileCSVReader,
    MultiSensorFusion,
    OrdinaryKriging3D,
    ProfileBuilder,
    SerialCSVReader,
    write_profile_nc,
)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="recorded CSV stream to replay")
    src.add_argument("--port", help="serial port, e.g. /dev/ttyUSB0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--duration", type=float, default=None,
                   help="seconds to read from the serial port")
    p.add_argument("--output", default="plume_profile.nc")
    p.add_argument("--grid", type=int, nargs=3, metavar=("NX", "NY", "NZ"),
                   default=(24, 24, 16))
    p.add_argument("--dt", type=float, default=1.0,
                   help="alignment time-grid step in seconds")
    p.add_argument("--gate", type=float, default=9.0,
                   help="innovation chi-square gate for outlier rejection")
    p.add_argument("--kriging-range", type=float, default=8.0,
                   help="variogram range in metres")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.input:
        reader = FileCSVReader(args.input)
        samples_iter = reader.samples()
    else:
        reader = SerialCSVReader(args.port, baudrate=args.baud)
        samples_iter = reader.samples(duration=args.duration)

    samples = list(samples_iter)
    if not samples:
        print("no valid samples received", file=sys.stderr)
        return 1
    print(f"ingested {len(samples)} samples "
          f"({reader.bad_lines} unparseable lines)")

    fusion = MultiSensorFusion(gate_chi2=args.gate)
    fusion.ingest_all(samples)
    for name, st in sorted(fusion.stats.items()):
        print(f"  {name:12s} accepted={st.accepted:5d} rejected={st.rejected}")

    t0, t1 = fusion.time_span()
    grid_times = np.arange(t0, t1 + 1e-9, args.dt)
    fused = fusion.align(grid_times)
    print(f"aligned {len(grid_times)} epochs over {t1 - t0:.1f} s, "
          f"channels: {', '.join(fused.channel_names())}")

    builder = ProfileBuilder(
        grid_shape=tuple(args.grid),
        kriging=OrdinaryKriging3D(range=args.kriging_range),
    )
    profile = builder.build(fused, samples)

    extra = {}
    if fused.anomaly_index is not None:
        from hydrofusion.profile_builder import ProfileBuilder as _PB
        pts = _PB.positions_at(samples, fused.times)
        from scipy.spatial import cKDTree
        xs, ys, zs = profile.x, profile.y, profile.z
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        targets = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
        est, _ = builder.kriging.predict(pts, fused.anomaly_index, targets)
        extra["anomaly_index"] = est.reshape(len(xs), len(ys), len(zs)).transpose(2, 1, 0)

    write_profile_nc(
        args.output,
        profile,
        extra_fields=extra,
        attributes={
            "time_coverage_start": f"{t0:.3f}",
            "time_coverage_end": f"{t1:.3f}",
            "alignment_dt_seconds": f"{args.dt}",
            "outlier_gate_chi2": f"{args.gate}",
        },
    )
    print(f"wrote {args.output} with fields: {', '.join(profile.fields)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
