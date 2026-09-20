"""Regression tests for evaluate._save_telemetry_csv.

The CSV must have a uniform column count so pandas / numpy loaders
don't break on ragged rows. The run summary lives in a sidecar .txt
file (audit item I5).
"""
import csv
import tempfile
import unittest
from pathlib import Path

from aircraft import Aircraft
from evaluate import _save_telemetry_csv
from mavlink_if import FlyingBoatVehicle
from ocean_real import load_buoy_default


class TelemetryCsvTests(unittest.TestCase):
    def test_uniform_columns_and_sidecar_summary(self):
        with tempfile.TemporaryDirectory() as td:
            sea = load_buoy_default()
            veh = FlyingBoatVehicle(Aircraft(), sea)
            veh.reset()
            veh.arm()
            veh.step()  # produce at least one _msg_log entry
            csv_path = Path(td) / "telemetry.csv"
            _save_telemetry_csv(veh, str(csv_path))

            with csv_path.open(newline="") as f:
                rows = list(csv.reader(f))
            self.assertGreater(len(rows), 1)
            ncols_header = len(rows[0])
            self.assertEqual(ncols_header, 12)
            for i, row in enumerate(rows[1:], start=2):
                self.assertEqual(
                    len(row), ncols_header,
                    f"row {i} has {len(row)} columns, expected {ncols_header}",
                )

            sidecar = csv_path.with_suffix(".summary.txt")
            self.assertTrue(sidecar.exists(),
                            "summary sidecar was not written")
            text = sidecar.read_text()
            self.assertIn("end_t=", text)
            self.assertIn("final_z=", text)
            self.assertIn("final_x=", text)


if __name__ == "__main__":
    unittest.main()
