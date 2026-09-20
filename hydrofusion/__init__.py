"""Hydrothermal plume multi-sensor fusion toolkit for ROV surveys."""

from .serial_reader import SensorSample, SerialCSVReader, FileCSVReader
from .kalman_fusion import KalmanFilter1D, MultiSensorFusion, FusedSeries
from .profile_builder import ProfileBuilder, OrdinaryKriging3D, ProfileResult
from .netcdf_writer import write_profile_nc

__version__ = "0.1.0"

__all__ = [
    "SensorSample",
    "SerialCSVReader",
    "FileCSVReader",
    "KalmanFilter1D",
    "MultiSensorFusion",
    "FusedSeries",
    "ProfileBuilder",
    "OrdinaryKriging3D",
    "ProfileResult",
    "write_profile_nc",
]
