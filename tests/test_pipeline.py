"""Unit tests for the hydrothermal fusion pipeline (no hardware needed)."""

import os
import tempfile
import unittest

import numpy as np

from hydrothermal_fusion.kalman_fusion import (KalmanFilter1D, KalmanFusion,
                                               hampel_mask)
from hydrothermal_fusion.netcdf_writer import write_profile_nc
from hydrothermal_fusion.profile_builder import (OrdinaryKriging3D,
                                                 build_profile)
from hydrothermal_fusion.serial_reader import (MultiChannelCollector,
                                               parse_csv_line)


class TestSerialReader(unittest.TestCase):
    def test_parse_single_value(self):
        s = parse_csv_line("h2s,1700000000.5,143.2")
        self.assertEqual(s.channel, "h2s")
        self.assertAlmostEqual(s.timestamp, 1700000000.5)
        self.assertEqual(s.values, (143.2,))

    def test_parse_nav_vector(self):
        s = parse_csv_line("nav,1700000000.0,1.0,2.0,-1500.0")
        self.assertEqual(len(s.values), 3)

    def test_comments_and_blanks(self):
        self.assertIsNone(parse_csv_line("# header"))
        self.assertIsNone(parse_csv_line("   "))

    def test_malformed_raises(self):
        with self.assertRaises(ValueError):
            parse_csv_line("h2s,not_a_number")

    def test_collector_sorts_by_time(self):
        c = MultiChannelCollector()
        for line in ["ph,3.0,7.8", "ph,1.0,7.9", "ph,2.0,7.85"]:
            c.add(parse_csv_line(line))
        ts, vals = c.to_arrays()["ph"]
        self.assertTrue(np.all(np.diff(ts) > 0))
        self.assertAlmostEqual(vals[0], 7.9)


class TestKalmanFusion(unittest.TestCase):
    def test_hampel_flags_spikes(self):
        y = np.ones(50)
        y[25] = 100.0
        mask = hampel_mask(y, window=7, n_sigmas=4.0)
        self.assertFalse(mask[25])
        self.assertEqual(mask.sum(), 49)

    def test_filter_tracks_and_gates(self):
        rng = np.random.default_rng(0)
        t = np.arange(0.0, 100.0, 0.5)
        truth = np.sin(t / 10.0)
        y = truth + rng.normal(0, 0.05, t.shape)
        y[80] = 50.0  # gross outlier
        kf = KalmanFilter1D(process_noise=1e-3, measurement_noise=2.5e-3)
        filt, std = kf.filter(t, y)
        self.assertEqual(kf.rejected, 1)
        self.assertLess(np.sqrt(np.mean((filt - truth) ** 2)), 0.05)
        self.assertTrue(np.all(std > 0))

    def test_fusion_aligns_channels(self):
        rng = np.random.default_rng(1)
        t_fast = np.arange(0.0, 60.0, 0.25)
        t_slow = np.arange(0.0, 60.0, 1.0)
        data = {
            "temperature": (t_fast, 2.0 + rng.normal(0, 0.05, t_fast.shape)),
            "ph": (t_slow, 7.8 + rng.normal(0, 0.02, t_slow.shape)),
        }
        nav = (t_fast, np.column_stack([t_fast, t_fast * 0.0, t_fast * 0.0]))
        result = KalmanFusion(output_rate_hz=1.0).fuse(data, nav_data=nav)
        self.assertEqual(result.time.shape, result.channels["ph"].values.shape)
        self.assertEqual(result.nav.shape, (result.time.size, 3))
        self.assertAlmostEqual(
            np.nanmean(result.channels["temperature"].values), 2.0, places=1)

    def test_fusion_marks_long_gaps_nan(self):
        t = np.concatenate([np.arange(0.0, 10.0, 1.0),
                            np.arange(60.0, 70.0, 1.0)])
        data = {"ph": (t, np.full(t.shape, 7.8))}
        result = KalmanFusion(output_rate_hz=1.0, max_gap=5.0).fuse(data)
        mid = (result.time > 20.0) & (result.time < 50.0)
        self.assertTrue(np.all(np.isnan(result.channels["ph"].values[mid])))


class TestProfileBuilder(unittest.TestCase):
    def test_kriging_recovers_constant_field(self):
        rng = np.random.default_rng(2)
        pts = rng.uniform(0, 100, (60, 3))
        vals = np.full(60, 42.0)
        targets = rng.uniform(0, 100, (20, 3))
        krig = OrdinaryKriging3D(range_=30.0)
        est, var = krig.predict(pts, vals, targets)
        np.testing.assert_allclose(est, 42.0, atol=1e-6)
        self.assertTrue(np.all(var >= 0))

    def test_kriging_exact_at_data_locations(self):
        rng = np.random.default_rng(3)
        pts = rng.uniform(0, 100, (30, 3))
        vals = pts[:, 0] * 0.1
        krig = OrdinaryKriging3D(range_=40.0)
        est, var = krig.predict(pts, vals, pts.copy())
        np.testing.assert_allclose(est, vals, atol=1e-4)
        self.assertTrue(np.all(var < 1e-6))

    def test_build_profile_shapes(self):
        rng = np.random.default_rng(4)
        pos = rng.uniform(0, 50, (200, 3))
        vals = np.sin(pos[:, 0] / 10.0)
        grid = build_profile(pos, vals, "test", grid_shape=(8, 6, 4),
                             margin=2.0)
        self.assertEqual(grid.values.shape, (4, 6, 8))  # (nz, ny, nx)
        self.assertEqual(grid.variance.shape, (4, 6, 8))
        self.assertEqual(grid.x.size, 8)

    def test_build_profile_rejects_too_few_samples(self):
        with self.assertRaises(ValueError):
            build_profile(np.zeros((3, 3)), np.zeros(3), "tiny")


class TestNetCDFWriter(unittest.TestCase):
    def test_round_trip(self):
        from netCDF4 import Dataset
        rng = np.random.default_rng(5)
        pos = rng.uniform(0, 50, (100, 3))
        grid = build_profile(pos, rng.normal(0, 1, 100), "h2s",
                             grid_shape=(6, 5, 4), margin=1.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.nc")
            write_profile_nc(path, {"h2s": grid})
            with Dataset(path) as nc:
                self.assertIn("h2s", nc.variables)
                self.assertIn("h2s_kriging_variance", nc.variables)
                self.assertEqual(nc.dimensions["x"].size, 6)
                self.assertEqual(nc.getncattr("Conventions"), "CF-1.8")
                self.assertEqual(nc["h2s"].shape, (4, 5, 6))


if __name__ == "__main__":
    unittest.main()
