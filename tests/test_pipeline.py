"""End-to-end and unit tests for the hydrothermal fusion pipeline."""

import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hydrofusion import (  # noqa: E402
    FileCSVReader,
    KalmanFilter1D,
    MultiSensorFusion,
    OrdinaryKriging3D,
    ProfileBuilder,
    SensorSample,
    write_profile_nc,
)
from hydrofusion.serial_reader import parse_csv_line  # noqa: E402


class TestSerialReader(unittest.TestCase):
    def test_parse_epoch_and_iso(self):
        s = parse_csv_line("1726.5,H2S,12.3,1.0,2.0,-100.0")
        self.assertEqual(s.sensor, "h2s")
        self.assertAlmostEqual(s.value, 12.3)
        self.assertTrue(s.has_position)
        s2 = parse_csv_line("2026-09-20T10:00:00Z,pH,5.1")
        self.assertIsNotNone(s2)
        self.assertFalse(s2.has_position)

    def test_rejects_garbage(self):
        for bad in ("", "# comment", "1,2", "a,b,c", "1,ph,nan"):
            self.assertIsNone(parse_csv_line(bad))


class TestKalmanFusion(unittest.TestCase):
    def test_filter_tracks_constant(self):
        kf = KalmanFilter1D(process_noise=1e-3, measurement_noise=0.01,
                            initial_value=0.0)
        for _ in range(200):
            kf.predict(0.1)
            kf.update(5.0)
        self.assertAlmostEqual(kf.value, 5.0, places=2)

    def test_outlier_gating(self):
        fusion = MultiSensorFusion(gate_chi2=9.0)
        t = 0.0
        for i in range(50):
            fusion.ingest(SensorSample(t + i * 0.5, "h2s", 10.0 + 0.01 * i))
        # Massive spike must be rejected.
        self.assertFalse(fusion.ingest(SensorSample(t + 25.0, "h2s", 500.0)))
        self.assertEqual(fusion.stats["h2s"].rejected, 1)
        # And the estimate must stay near 10, not jump toward 500.
        fused = fusion.align(np.array([t + 25.0]))
        self.assertLess(abs(fused.channels["h2s"][0] - 10.5), 1.0)

    def test_multirate_alignment(self):
        fusion = MultiSensorFusion()
        t = np.arange(0.0, 20.0, 0.05)
        rng = np.random.default_rng(0)
        for ti in t:
            if int(ti * 100) % 100 == 0:      # 1 Hz pH
                fusion.ingest(SensorSample(ti, "ph", 7.0 + rng.normal(0, 0.05)))
            if int(ti * 100) % 20 == 0:       # 5 Hz temperature
                fusion.ingest(SensorSample(ti, "temperature", 4.0 + rng.normal(0, 0.1)))
        grid = np.arange(0.0, 20.0, 0.5)
        fused = fusion.align(grid)
        self.assertEqual(fused.channels["ph"].shape, grid.shape)
        self.assertEqual(fused.channels["temperature"].shape, grid.shape)
        self.assertAlmostEqual(np.mean(fused.channels["ph"]), 7.0, delta=0.1)
        self.assertIsNotNone(fused.anomaly_index)


class TestProfileBuilder(unittest.TestCase):
    def test_kriging_recovers_smooth_field(self):
        rng = np.random.default_rng(1)
        pts = rng.uniform(-10, 10, size=(120, 3))
        truth = np.exp(-0.5 * np.sum(pts ** 2, axis=1) / 25.0)
        noisy = truth + rng.normal(0, 0.02, size=len(pts))
        krig = OrdinaryKriging3D(range=8.0, sill=float(np.var(noisy)) + 1e-3)
        targets = rng.uniform(-10, 10, size=(40, 3))
        est, var = krig.predict(pts, noisy, targets)
        expected = np.exp(-0.5 * np.sum(targets ** 2, axis=1) / 25.0)
        self.assertLess(np.sqrt(np.mean((est - expected) ** 2)), 0.15)
        self.assertTrue(np.all(var >= 0.0))

    def test_build_profile_shapes(self):
        fusion = MultiSensorFusion()
        rng = np.random.default_rng(2)
        samples = []
        for i in range(300):
            t = i * 0.5
            x, y, z = rng.uniform(-10, 10, 3)
            samples.append(SensorSample(t, "h2s", 10 + x + rng.normal(0, 0.1), x, y, z))
        fusion.ingest_all(samples)
        grid = np.arange(0.0, 150.0, 1.0)
        fused = fusion.align(grid)
        builder = ProfileBuilder(grid_shape=(8, 8, 6))
        profile = builder.build(fused, samples)
        self.assertEqual(profile.fields["h2s"].shape, (6, 8, 8))
        self.assertEqual(len(profile.x), 8)


class TestNetCDFWriter(unittest.TestCase):
    def test_roundtrip(self):
        from netCDF4 import Dataset
        fusion = MultiSensorFusion()
        rng = np.random.default_rng(3)
        samples = []
        for i in range(200):
            t = i * 0.5
            x, y, z = rng.uniform(-5, 5, 3)
            samples.append(SensorSample(t, "temperature", 5 + y, x, y, z))
        fusion.ingest_all(samples)
        fused = fusion.align(np.arange(0.0, 100.0, 1.0))
        profile = ProfileBuilder(grid_shape=(6, 6, 4)).build(fused, samples)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.nc")
            write_profile_nc(path, profile, attributes={"cruise": "TEST01"})
            with Dataset(path) as ds:
                self.assertIn("temperature", ds.variables)
                self.assertEqual(ds.dimensions["x"].size, 6)
                self.assertEqual(ds.getncattr("cruise"), "TEST01")


class TestEndToEnd(unittest.TestCase):
    def test_cli_pipeline(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "stream.csv")
            nc_path = os.path.join(tmp, "plume.nc")
            env = dict(os.environ, PYTHONPATH=root)
            subprocess.run(
                [sys.executable, "scripts/simulate_rov_stream.py",
                 "--output", csv_path, "--duration", "60"],
                cwd=root, env=env, check=True, capture_output=True)
            result = subprocess.run(
                [sys.executable, "main.py", "--input", csv_path,
                 "--output", nc_path, "--grid", "8", "8", "6"],
                cwd=root, env=env, check=True, capture_output=True, text=True)
            self.assertTrue(os.path.exists(nc_path))
            self.assertIn("wrote", result.stdout)


if __name__ == "__main__":
    unittest.main()
