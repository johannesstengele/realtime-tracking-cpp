import copy
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from evaluation.analysis import (analyze, bootstrap_mean_interval, calibration_from_frames,
                                 read_frames, reference_summary, run_metrics)


EXECUTABLE = Path(os.environ["TRACKING_EXPERIMENT_EXECUTABLE"]).resolve() if os.environ.get("TRACKING_EXPERIMENT_EXECUTABLE") else None
if __name__ == "__main__" and len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
    EXECUTABLE = Path(sys.argv.pop(1)).resolve()


class EvaluationStatisticsTest(unittest.TestCase):
    def test_bootstrap_keeps_runs_as_independent_sampling_units(self):
        interval = bootstrap_mean_interval([0, 1], 2000)
        self.assertEqual(interval, [0, 1])
        self.assertEqual(bootstrap_mean_interval([2, 2], 100), [2, 2])

    def test_reports_missing_estimates_and_false_detections(self):
        row = {"visible": 0, "detected": 1, "estimated": 1, "measurement_squared_error": 100,
               "position_squared_error": 100, "velocity_squared_error": 25,
               "estimate_x": 10, "true_x": 0, "estimate_y": 0, "true_y": 0,
               "position_nees": 100, "nees": 120, "nis": None, "track_id": 1, "status": "tracking"}
        missing = copy.deepcopy(row)
        missing.update(visible=1, detected=0, estimated=0, status="lost")
        result = run_metrics([row, missing])
        self.assertEqual(result["correct_association_rate"], 0)
        self.assertEqual(result["wrong_association_rate"], 0.5)
        self.assertEqual(result["estimate_availability"], 0.5)
        self.assertEqual(result["position_rmse"], 10)
        self.assertEqual(result["position_95_coverage"], 0)
        self.assertEqual(result["recovery_delays_frames"], [None])


@unittest.skipUnless(EXECUTABLE, "C++ experiment executable is required")
class ExperimentInterfaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.raw = cls.root / "sequence"
        cls.command(cls.raw, "--scenario", "long_dropout", "--frames", "60", "--runs", "2", "--seed", "20000")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @staticmethod
    def command(directory, *options):
        return subprocess.run([str(EXECUTABLE), "--output", str(directory), *options],
                              check=True, capture_output=True, text=True)

    def test_exports_missing_values_and_lifecycle_without_column_shift(self):
        configuration, rows, sequences = read_frames(self.raw)
        self.assertEqual(configuration["runs_per_scenario"], 2)
        self.assertEqual(len(rows), 120)
        self.assertEqual(len(sequences), 2)
        lost = next(row for row in rows if row["status"] == "lost")
        self.assertIsNone(lost["estimate_x"])
        self.assertIsNone(lost["nis"])
        self.assertGreater(lost["pipeline_ns"], 0)
        summary = analyze(self.raw, 100)
        self.assertTrue(summary["scenarios"]["long_dropout"]["lifecycle"]["long_identity_reinitialized"])

    def test_preserves_full_width_seed_and_reproducible_numerical_outputs(self):
        seed = 18446744073709551600
        directories = [self.root / "repeat-a", self.root / "repeat-b"]
        data = []
        for directory in directories:
            self.command(directory, "--scenario", "nominal", "--frames", "12", "--runs", "1", "--seed", str(seed))
            _, rows, _ = read_frames(directory)
            self.assertEqual(rows[0]["seed"], seed)
            data.append([{key: value for key, value in row.items() if not key.endswith("_ns")} for row in rows])
        self.assertEqual(data[0], data[1])
        calibration = calibration_from_frames(directories[0])
        self.assertGreaterEqual(calibration["measurement_variance"], 0.04)

    def test_rejects_invalid_options_and_preserves_existing_output(self):
        before = (self.raw / "frames.csv").read_bytes()
        with self.assertRaises(subprocess.CalledProcessError):
            self.command(self.raw, "--runs", "1")
        self.assertEqual(before, (self.raw / "frames.csv").read_bytes())
        for arguments in (("--runs", "-1"), ("--scenario", "unknown"), ("--measurement-variance", "nan")):
            with self.assertRaises(subprocess.CalledProcessError):
                self.command(self.root / "invalid", *arguments)

    def test_detects_truncated_csv(self):
        directory = self.root / "truncated"
        directory.mkdir()
        (directory / "configuration.json").write_bytes((self.raw / "configuration.json").read_bytes())
        (directory / "frames.csv").write_text("scenario,seed\nnominal,1\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            read_frames(directory)

    def test_reference_is_ungated_and_uses_independent_trials(self):
        directory = self.root / "reference"
        self.command(directory, "--scenario", "reference", "--runs", "1000", "--frames", "20", "--seed", "800000")
        reference = reference_summary(directory)
        self.assertEqual(reference["trials"], 1000)
        self.assertTrue(reference["passed"])
        self.assertAlmostEqual(reference["nis"]["mean"], 2, delta=0.25)
        self.assertAlmostEqual(reference["nees"]["mean"], 4, delta=0.4)


if __name__ == "__main__":
    unittest.main()
