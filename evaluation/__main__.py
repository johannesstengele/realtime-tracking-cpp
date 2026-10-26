import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

from .analysis import analyze, calibration_from_frames, reference_summary


def run_process(executable, output, runs, frames, seed, scenario, variance=0.36):
    command = [str(executable), "--output", str(output), "--runs", str(runs), "--frames", str(frames),
               "--seed", str(seed), "--scenario", scenario, "--warmup", str(min(10, frames - 1)),
               "--measurement-variance", str(variance)]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def revision():
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def main():
    parser = argparse.ArgumentParser(description="Calibrate and evaluate the C++ tracking pipeline")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--executable", type=Path)
    mode.add_argument("--input", type=Path, help="Analyze an existing validation raw-data directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--calibration-seed", type=int, default=100)
    parser.add_argument("--calibration-runs", type=int, default=20)
    parser.add_argument("--reference-trials", type=int, default=10000)
    parser.add_argument("--reference-seed", type=int, default=1000000)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--budget-ms", type=float, default=1000 / 30)
    parser.add_argument("--strict", action="store_true", help="Return failure if full delivery acceptance is not met")
    arguments = parser.parse_args()
    if arguments.runs < 1 or arguments.frames < 1 or arguments.bootstrap < 100 or arguments.reference_trials < 100:
        parser.error("Positive runs/frames, at least 100 bootstrap repetitions and 100 reference trials are required")
    import math
    if not math.isfinite(arguments.budget_ms) or arguments.budget_ms <= 0:
        parser.error("Latency budget must be finite and positive")
    arguments.output.mkdir(parents=True, exist_ok=True)
    report_path = arguments.output / "report.json"
    if report_path.exists():
        parser.error("Report already exists; choose a new output directory")
    calibration = None
    reference = None
    try:
        if arguments.executable:
            executable = arguments.executable.resolve(strict=True)
            if arguments.calibration_runs < 1:
                parser.error("Calibration run count must be positive")
            calibration_seeds = range(arguments.calibration_seed, arguments.calibration_seed + arguments.calibration_runs)
            validation_seeds = range(arguments.seed, arguments.seed + arguments.runs)
            if max(calibration_seeds.start, validation_seeds.start) < min(calibration_seeds.stop, validation_seeds.stop):
                parser.error("Calibration and validation seed ranges must be disjoint")
            calibration_path = arguments.output / "calibration"
            run_process(executable, calibration_path, arguments.calibration_runs, arguments.frames,
                        arguments.calibration_seed, "nominal")
            calibration = calibration_from_frames(calibration_path)
            validation_path = arguments.output / "validation"
            run_process(executable, validation_path, arguments.runs, arguments.frames,
                        arguments.seed, "all", calibration["measurement_variance"])
            reference_path = arguments.output / "reference"
            run_process(executable, reference_path, arguments.reference_trials, 60, arguments.reference_seed, "reference")
            reference = reference_summary(reference_path)
        else:
            validation_path = arguments.input
        report = analyze(validation_path, arguments.bootstrap, arguments.budget_ms)
        report["source_revision"] = revision()
        report["machine"] = {"system": platform.system(), "release": platform.release(),
                             "architecture": platform.machine(), "processor": platform.processor(),
                             "python": platform.python_version()}
        report["calibration"] = calibration
        report["linear_gaussian_reference"] = reference
        report["evidence_complete"] = (set(report["scenarios"]) == {
            "nominal", "maneuver", "clutter", "short_occlusion", "long_dropout", "degraded", "boundary"}
            and report["configuration"]["runs_per_scenario"] >= 100
            and report["configuration"]["frames_per_run"] >= 120
            and report["configuration"]["build_configuration"] == "Release"
            and calibration is not None and reference is not None and reference["passed"])
        report["passed"] = bool(report["passed"] and report["evidence_complete"])
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for name, result in report["scenarios"].items():
            metrics = result["metrics"]
            print(f"{name}: correct={metrics['correct_association_rate']['mean']:.4f}, "
                  f"position_rmse={metrics['position_rmse']['mean']:.4f}px, "
                  f"pipeline_p99={result['timings']['pipeline_ms']['p99']:.3f}ms, "
                  f"passed={all(result['acceptance'].values())}")
        print(f"Complete acceptance: {report['passed']}")
        return 1 if arguments.strict and not report["passed"] else 0
    except subprocess.CalledProcessError as error:
        print(error.stderr or str(error), file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
