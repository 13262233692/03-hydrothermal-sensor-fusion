"""Write gridded hydrothermal plume profiles to NetCDF (CF-flavoured)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
from netCDF4 import Dataset

from .profile_builder import ProfileResult

# Canonical metadata for known channels.
CHANNEL_META: Dict[str, Dict[str, str]] = {
    "ph":          {"units": "1",       "long_name": "seawater pH (total scale)"},
    "h2s":         {"units": "umol/kg", "long_name": "dissolved hydrogen sulfide concentration"},
    "temperature": {"units": "degC",    "long_name": "seawater temperature"},
    "turbidity":   {"units": "NTU",     "long_name": "turbidity"},
    "anomaly_index": {"units": "1",     "long_name": "weighted multi-sensor hydrothermal anomaly index"},
}

_FILL = -9999.0


def write_profile_nc(
    path: str,
    profile: ProfileResult,
    extra_fields: Optional[Dict[str, np.ndarray]] = None,
    attributes: Optional[Dict[str, str]] = None,
) -> str:
    """Write a :class:`ProfileResult` (plus optional extra grids) to NetCDF.

    Dimensions: ``(z, y, x)``. Every channel becomes a float32 variable with
    a ``_FillValue`` where the grid holds NaN; kriging variance grids are
    stored as ``<name>_variance`` when present.
    """
    fields = dict(profile.fields)
    if extra_fields:
        fields.update(extra_fields)

    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("x", len(profile.x))
        ds.createDimension("y", len(profile.y))
        ds.createDimension("z", len(profile.z))

        for name, values, units, positive in (
            ("x", profile.x, "m", None),
            ("y", profile.y, "m", None),
            ("z", profile.z, "m", "up"),
        ):
            var = ds.createVariable(name, "f8", (name,))
            var[:] = values
            var.units = units
            var.axis = name.upper()
            if positive:
                var.positive = positive

        for name, grid in fields.items():
            meta = CHANNEL_META.get(name, {"units": "unknown", "long_name": name})
            var = ds.createVariable(
                name, "f4", ("z", "y", "x"), fill_value=np.float32(_FILL), zlib=True
            )
            var[:] = np.ma.masked_invalid(np.asarray(grid, dtype=np.float32))
            var.units = meta["units"]
            var.long_name = meta["long_name"]
            var.coordinates = "z y x"

            if name in profile.variances:
                vvar = ds.createVariable(
                    f"{name}_variance", "f4", ("z", "y", "x"),
                    fill_value=np.float32(_FILL), zlib=True,
                )
                vvar[:] = np.ma.masked_invalid(
                    np.asarray(profile.variances[name], dtype=np.float32)
                )
                vvar.long_name = f"kriging variance of {name}"
                vvar.coordinates = "z y x"

        ds.title = "Hydrothermal plume 3-D concentration profile"
        ds.institution = "ROV hydrothermal survey"
        ds.source = "Kalman-fused multi-sensor ROV telemetry, ordinary-kriging gridded"
        ds.history = f"created {datetime.now(timezone.utc).isoformat()}"
        ds.Conventions = "CF-1.8"
        for key, value in (attributes or {}).items():
            setattr(ds, key, value)

    return path
