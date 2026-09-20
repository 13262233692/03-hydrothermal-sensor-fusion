"""CF-compliant NetCDF-4 output for reconstructed plume profiles.

The product holds one 3-D variable per fused channel (plus its Kriging
variance) on a shared regular ``(z, y, x)`` grid, together with the fused
1-D time series for traceability.  Metadata follows CF-1.8 conventions
so the file opens cleanly in Panoply, Ferret, xarray, etc.
"""

from __future__ import annotations

import datetime as _dt
import logging
import platform
from typing import Dict, Optional

import numpy as np
from netCDF4 import Dataset, default_fillvals

from .profile_builder import ProfileGrid

log = logging.getLogger(__name__)

#: (units, standard_name, long_name) per channel.
CHANNEL_META = {
    "ph": ("1", "sea_water_ph_reported_on_total_scale", "pH (total scale)"),
    "h2s": ("umol/kg", "concentration_of_h2s_in_sea_water",
            "dissolved hydrogen sulfide concentration"),
    "temperature": ("degC", "sea_water_temperature", "in-situ temperature"),
    "turbidity": ("NTU", "sea_water_turbidity", "optical turbidity"),
}


def _channel_meta(name: str):
    return CHANNEL_META.get(name, ("1", "", name))


def write_profile_nc(path: str, grids: Dict[str, ProfileGrid],
                     fusion_result=None,
                     title: str = "ROV hydrothermal plume 3-D reconstruction",
                     institution: str = "Hydrothermal Vent Survey Group",
                     source: str = "ROV multi-sensor Kalman fusion",
                     extra_attrs: Optional[dict] = None) -> str:
    """Write interpolated profiles (and optional fused series) to NetCDF."""
    if not grids:
        raise ValueError("no grids supplied")
    first = next(iter(grids.values()))
    xs, ys, zs = first.x, first.y, first.z

    with Dataset(path, "w", format="NETCDF4") as nc:
        nc.createDimension("x", xs.size)
        nc.createDimension("y", ys.size)
        nc.createDimension("z", zs.size)

        x_var = nc.createVariable("x", "f8", ("x",))
        x_var.units = "m"
        x_var.long_name = "easting, local survey frame"
        x_var[:] = xs
        y_var = nc.createVariable("y", "f8", ("y",))
        y_var.units = "m"
        y_var.long_name = "northing, local survey frame"
        y_var[:] = ys
        z_var = nc.createVariable("z", "f8", ("z",))
        z_var.units = "m"
        z_var.positive = "up"
        z_var.long_name = "height above local datum"
        z_var[:] = zs

        for name, grid in sorted(grids.items()):
            units, std_name, long_name = _channel_meta(name)
            var = nc.createVariable(
                name, "f4", ("z", "y", "x"), zlib=True, complevel=4,
                fill_value=default_fillvals["f4"])
            var.units = units
            var.long_name = f"{long_name}, kriged 3-D field"
            if std_name:
                var.standard_name = std_name
            var.coordinates = "z y x"
            var.comment = (f"ordinary kriging of {grid.n_samples} "
                           "georeferenced fused samples")
            var[:] = grid.values.astype("f4")

            kv = nc.createVariable(
                f"{name}_kriging_variance", "f4", ("z", "y", "x"),
                zlib=True, complevel=4, fill_value=default_fillvals["f4"])
            kv.units = f"({units})^2"
            kv.long_name = f"kriging variance of {long_name}"
            kv.coordinates = "z y x"
            kv[:] = grid.variance.astype("f4")

        if fusion_result is not None:
            _write_fused_series(nc, fusion_result)

        nc.title = title
        nc.institution = institution
        nc.source = source
        nc.Conventions = "CF-1.8"
        nc.history = (f"created {_dt.datetime.utcnow().isoformat()}Z "
                      f"by hydrothermal_fusion on {platform.node()}")
        if extra_attrs:
            for key, value in extra_attrs.items():
                setattr(nc, key, str(value))
    log.info("wrote %s", path)
    return path


def _write_fused_series(nc: Dataset, fusion_result) -> None:
    """Attach the time-aligned fused 1-D series on a ``time`` dimension."""
    times = fusion_result.time
    nc.createDimension("time", times.size)
    t_var = nc.createVariable("time", "f8", ("time",))
    t_var.units = "seconds since 1970-01-01 00:00:00 UTC"
    t_var.calendar = "proleptic_gregorian"
    t_var.standard_name = "time"
    t_var[:] = times

    for name, channel in sorted(fusion_result.channels.items()):
        units, std_name, long_name = _channel_meta(name)
        var = nc.createVariable(f"{name}_fused", "f4", ("time",),
                                zlib=True, complevel=4,
                                fill_value=default_fillvals["f4"])
        var.units = units
        var.long_name = f"{long_name}, Kalman-fused time series"
        if std_name:
            var.standard_name = std_name
        var.ancillary_variables = f"{name}_fused_std"
        var[:] = np.ma.masked_invalid(channel.values.astype("f4"))

        s_var = nc.createVariable(f"{name}_fused_std", "f4", ("time",),
                                  zlib=True, complevel=4,
                                  fill_value=default_fillvals["f4"])
        s_var.units = units
        s_var.long_name = f"1-sigma uncertainty of {long_name}"
        s_var[:] = np.ma.masked_invalid(channel.std.astype("f4"))

    if fusion_result.nav is not None:
        for col, axis in enumerate("xyz"):
            nav_var = nc.createVariable(f"rov_{axis}", "f4", ("time",),
                                        zlib=True, complevel=4,
                                        fill_value=default_fillvals["f4"])
            nav_var.units = "m"
            nav_var.long_name = f"ROV {axis} position, local survey frame"
            nav_var[:] = np.ma.masked_invalid(
                fusion_result.nav[:, col].astype("f4"))
