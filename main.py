#!/usr/bin/env python3
"""ROV hydrothermal plume multi-sensor fusion pipeline.

Subcommands
-----------
simulate   Generate synthetic per-port CSV streams (hardware-free test).
fuse       Read CSV streams (files or serial ports), run Kalman fusion,
           build 3-D Kriged profiles and write a NetCDF product.

Examples
--------
    python3 main.py simulate --out data/sim --duration 600
    python3 main.py fuse --input data/sim --out plume.nc
    python3 main.py fuse --ports /dev/ttyUSB0 /dev/ttyUSB1 --out plume.nc
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import queue
import sys
import time

from hydrothermal_fusion.kalman_fusion import KalmanFusion
from hydrothermal_fusion.netcdf_writer import write_profile_nc
from hydrothermal_fusion.profile_builder import build_all_profiles
from hydrothermal_fusion.serial_reader import (NAV_CHANNEL,
                                               CSVStreamReader,
                                               MultiChannelCollector,
                                               SerialPortReader)

log = logging.getLogger("hydrothermal_fusion")


def cmd_simulate(args: argparse.Namespace) -> int:
    from hydrothermal_fusion.sensor_simulator import simulate

    paths = simulate(args.out, duration=args.duration, seed=args.seed)
    for path in paths:
        print(f"wrote {path}")
    return 0


def _collect_from_files(inputs: list) -> MultiChannelCollector:
    collector = MultiChannelCollector()
    files = []
    for item in inputs:
        if os.path.isdir(item):
            files.extend(sorted(glob.glob(os.path.join(item, "*.csv"))))
        else:
            files.append(item)
    if not files:
        raise SystemExit("no CSV input files found")
    for path in files:
        with open(path, "r", encoding="ascii", errors="replace") as fh:
            reader = CSVStreamReader(fh, source=path)
            count = collector.add_from(reader.samples())
        log.info("read %s (%d samples, %d bad lines)",
                 path, count, reader.bad_lines)
    return collector


def _collect_from_serial(ports: list, baudrate: int,
                         duration: float) -> MultiChannelCollector:
    sample_queue: "queue.Queue" = queue.Queue()
    readers = [SerialPortReader(p, sample_queue, baudrate=baudrate)
               for p in ports]
    for reader in readers:
        reader.start()
    collector = MultiChannelCollector()
    deadline = time.time() + duration
    log.info("listening on %s for %.0f s ...", ", ".join(ports), duration)
    while time.time() < deadline:
        try:
            collector.add(sample_queue.get(timeout=0.5))
        except queue.Empty:
            pass
    for reader in readers:
        reader.stop()
    return collector


def cmd_fuse(args: argparse.Namespace) -> int:
    if args.ports:
        collector = _collect_from_serial(args.ports, args.baud,
                                         args.listen_seconds)
    else:
        collector = _collect_from_files(args.input)

    data = collector.to_arrays()
    log.info("channels acquired: %s",
             {k: v[0].size for k, v in data.items()})
    nav = data.pop(NAV_CHANNEL, None)

    fusion = KalmanFusion(output_rate_hz=args.rate, max_gap=args.max_gap)
    result = fusion.fuse(data, nav_data=nav)
    for name, ch in sorted(result.channels.items()):
        log.info("fused %-11s rejected: hampel=%d gate=%d",
                 name, ch.n_rejected_hampel, ch.n_rejected_gate)

    grids = build_all_profiles(result,
                               grid_shape=(args.nx, args.ny, args.nz),
                               margin=args.margin)
    write_profile_nc(args.out, grids, fusion_result=result,
                     extra_attrs={"rov": args.rov_name,
                                  "cruise": args.cruise})
    print(f"wrote {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ROV hydrothermal plume multi-sensor fusion pipeline")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="generate synthetic CSV streams")
    sim.add_argument("--out", default="data/sim",
                     help="output directory for CSV files")
    sim.add_argument("--duration", type=float, default=600.0,
                     help="survey duration in seconds")
    sim.add_argument("--seed", type=int, default=42)
    sim.set_defaults(func=cmd_simulate)

    fuse = sub.add_parser("fuse", help="fuse CSV streams into a NetCDF product")
    fuse.add_argument("--input", nargs="+", default=["data/sim"],
                      help="CSV files or directories (default: data/sim)")
    fuse.add_argument("--ports", nargs="+", default=None,
                      help="serial ports to read live, e.g. /dev/ttyUSB0")
    fuse.add_argument("--baud", type=int, default=115200)
    fuse.add_argument("--listen-seconds", type=float, default=60.0,
                      help="acquisition window when reading serial ports")
    fuse.add_argument("--rate", type=float, default=1.0,
                      help="common output time-grid rate (Hz)")
    fuse.add_argument("--max-gap", type=float, default=5.0,
                      help="max seconds to interpolate across data gaps")
    fuse.add_argument("--nx", type=int, default=24)
    fuse.add_argument("--ny", type=int, default=24)
    fuse.add_argument("--nz", type=int, default=12)
    fuse.add_argument("--margin", type=float, default=5.0,
                      help="grid margin around the survey box (m)")
    fuse.add_argument("--rov-name", default="ROV-1")
    fuse.add_argument("--cruise", default="unspecified")
    fuse.add_argument("--out", default="plume_profile.nc")
    fuse.set_defaults(func=cmd_fuse)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
