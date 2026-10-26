import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


SCENARIOS = ("nominal", "maneuver", "clutter", "short_occlusion", "long_dropout", "degraded", "boundary")
TIMINGS = ("render_ns", "prediction_ns", "detection_ns", "association_ns", "correction_ns", "tracker_ns", "pipeline_ns")


def quantile(values, probability):
    ordered = sorted(values)
    if not ordered:
        return None
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def read_frames(directory):
    directory = Path(directory)
    configuration = json.loads((directory / "configuration.json").read_text(encoding="utf-8"))
    with (directory / "frames.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = configuration["runs_per_scenario"] * configuration["frames_per_run"] * len(configuration["scenarios"])
    if len(rows) != expected or not rows:
        raise ValueError("Experiment row count does not match its configuration")
    sequences = defaultdict(list)
    numeric = ("time", "visible", "detected", "estimated", "timed",
               "true_x", "true_y", "true_vx", "true_vy", "measured_x", "measured_y",
               "estimate_x", "estimate_y", "estimate_vx", "estimate_vy", "position_squared_error",
               "velocity_squared_error", "measurement_squared_error", "position_nees", "nees", "nis") + TIMINGS
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Malformed experiment CSV")
        for key in numeric:
            if row[key]:
                row[key] = float(row[key])
                if not math.isfinite(row[key]):
                    raise ValueError("Nonfinite experiment value: " + key)
            else:
                row[key] = None
        for key in ("seed", "frame", "track_id"):
            row[key] = int(row[key])
        sequences[row["scenario"], row["seed"]].append(row)
    for scenario in configuration["scenarios"]:
        for seed in range(configuration["first_seed"], configuration["first_seed"] + configuration["runs_per_scenario"]):
            sequence = sequences[scenario, seed]
            if [row["frame"] for row in sequence] != list(range(configuration["frames_per_run"])):
                raise ValueError("Missing, duplicated or reordered frames")
    if len(sequences) != configuration["runs_per_scenario"] * len(configuration["scenarios"]):
        raise ValueError("Unexpected scenario or seed")
    return configuration, rows, sequences


def calibration_from_frames(directory):
    configuration, rows, _ = read_frames(directory)
    errors = [(row["measured_x"] - row["true_x"], row["measured_y"] - row["true_y"])
              for row in rows if row["scenario"] == "nominal" and row["visible"] and row["detected"]
              and row["measurement_squared_error"] <= 9]
    if len(errors) < 2:
        raise ValueError("Calibration needs at least two correct nominal measurements")
    x, y = zip(*errors)
    variance = max(statistics.variance(x), statistics.variance(y), 0.04)
    return {"first_seed": configuration["first_seed"], "runs": configuration["runs_per_scenario"],
            "samples": len(errors), "mean_error": [statistics.mean(x), statistics.mean(y)],
            "sample_variances": [statistics.variance(x), statistics.variance(y)],
            "measurement_variance": variance, "variance_floor": 0.04}


def run_metrics(rows):
    visible = [row for row in rows if row["visible"]]
    accepted = [row for row in rows if row["detected"]]
    correct = [row for row in visible if row["detected"] and row["measurement_squared_error"] <= 9]
    estimated = [row for row in rows if row["estimated"]]
    wrong = [row for row in accepted if not row["visible"] or row["measurement_squared_error"] > 9]
    errors = [row["position_squared_error"] for row in estimated]
    velocity_errors = [row["velocity_squared_error"] for row in estimated]
    innovation = [row["nis"] for row in rows if row["nis"] is not None]
    identities = [row["track_id"] for row in correct]
    unexpected_switches = sum(left != right for left, right in zip(identities, identities[1:]))
    recovery = []
    short_preserved = []
    long_reinitialized = []
    for index in range(1, len(rows)):
        if rows[index]["visible"] and not rows[index - 1]["visible"]:
            start = index - 1
            while start > 0 and not rows[start - 1]["visible"]:
                start -= 1
            previous = next((row["track_id"] for row in reversed(rows[:start]) if row["detected"]), None)
            returning = next((row for row in rows[index:] if row["visible"] and row["detected"]
                              and row["measurement_squared_error"] <= 9), None)
            recovery.append(None if returning is None else returning["frame"] - index)
            if index - start <= 5:
                short_preserved.append(returning is not None and returning["track_id"] == previous
                                       and all(row["estimated"] and row["track_id"] == previous
                                               for row in rows[start:index]))
            else:
                long_reinitialized.append(returning is not None and returning["track_id"] != previous
                                          and any(row["status"] == "lost" for row in rows[start:index]))
    count_visible = max(1, len(visible))
    count_all = len(rows)
    rmse = lambda values: math.sqrt(statistics.mean(values)) if values else None
    return {
        "correct_association_rate": len(correct) / count_visible,
        "wrong_association_rate": len(wrong) / count_all,
        "estimate_availability": len(estimated) / count_all,
        "position_rmse": rmse(errors), "velocity_rmse": rmse(velocity_errors),
        "measurement_rmse": rmse([row["measurement_squared_error"] for row in correct]),
        "correct_association_position_rmse": rmse([row["position_squared_error"] for row in correct]),
        "position_bias_x": statistics.mean(row["estimate_x"] - row["true_x"] for row in estimated) if estimated else None,
        "position_bias_y": statistics.mean(row["estimate_y"] - row["true_y"] for row in estimated) if estimated else None,
        "position_95_coverage": statistics.mean(row["position_nees"] <= 5.991464547107979 for row in estimated) if estimated else None,
        "mean_nees": statistics.mean(row["nees"] for row in estimated) if estimated else None,
        "mean_accepted_nis": statistics.mean(innovation) if innovation else None,
        "lost_events": sum(row["status"] == "lost" for row in rows),
        "identity_changes": unexpected_switches,
        "recovery_delays_frames": recovery, "short_identity_preserved": short_preserved,
        "long_identity_reinitialized": long_reinitialized,
    }


def bootstrap_mean_interval(values, repetitions=1000, seed=314159):
    if not values:
        return None
    generator = random.Random(seed)
    means = [sum(generator.choices(values, k=len(values))) / len(values) for _ in range(repetitions)]
    return [quantile(means, 0.025), quantile(means, 0.975)]


def analyze(directory, repetitions=1000, budget_ms=1000 / 30):
    configuration, rows, sequences = read_frames(directory)
    summary = {"configuration": configuration, "latency_budget_ms": budget_ms,
               "interval_method": "percentile bootstrap of independent run means, 95%, fixed seed 314159",
               "bootstrap_repetitions": repetitions, "scenarios": {}}
    for scenario in configuration["scenarios"]:
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        runs = [run_metrics(sequence) for (name, _), sequence in sequences.items() if name == scenario]
        metrics = {}
        for name in runs[0]:
            if isinstance(runs[0][name], list):
                continue
            values = [run[name] for run in runs if run[name] is not None]
            metrics[name] = {"mean": statistics.mean(values) if values else None,
                             "ci95": bootstrap_mean_interval(values, repetitions)}
        timings = {}
        for stage in TIMINGS:
            values = [row[stage] / 1e6 for row in scenario_rows if row["timed"]]
            timings[stage.removesuffix("_ns") + "_ms"] = {
                "samples": len(values), "mean": statistics.mean(values) if values else None,
                "median": quantile(values, 0.5), "p95": quantile(values, 0.95),
                "p99": quantile(values, 0.99), "maximum": max(values) if values else None,
                "standard_deviation": statistics.pstdev(values) if values else None,
                "deadline_misses": sum(value > budget_ms for value in values),
            }
        delays = [delay for run in runs for delay in run["recovery_delays_frames"]]
        lifecycle = {
            "return_events": len(delays), "unrecovered_events": sum(delay is None for delay in delays),
            "maximum_recovery_delay_frames": max((delay for delay in delays if delay is not None), default=None),
            "short_identity_preserved": all(value for run in runs for value in run["short_identity_preserved"]),
            "long_identity_reinitialized": all(value for run in runs for value in run["long_identity_reinitialized"]),
        }
        minimum_rate = 0.99 if scenario == "nominal" else 0.95
        maximum_error = 1.0 if scenario == "nominal" else 3.0
        accuracy = (metrics["correct_association_rate"]["mean"] >= minimum_rate
                    and metrics["wrong_association_rate"]["mean"] <= 0.01
                    and metrics["correct_association_position_rmse"]["mean"] is not None
                    and metrics["correct_association_position_rmse"]["mean"] <= maximum_error)
        continuity = (not lifecycle["unrecovered_events"] and lifecycle["short_identity_preserved"]
                      and lifecycle["long_identity_reinitialized"]
                      and all(delay is not None and delay <= 2 for delay in delays))
        latency = timings["pipeline_ms"]["p99"] is not None and timings["pipeline_ms"]["p99"] <= budget_ms
        summary["scenarios"][scenario] = {"runs": len(runs), "metrics": metrics, "timings": timings,
                                           "lifecycle": lifecycle,
                                           "acceptance": {"accuracy": accuracy, "lifecycle": continuity, "latency": latency}}
    summary["passed"] = all(all(result["acceptance"].values()) for result in summary["scenarios"].values())
    summary["raw_csv_sha256"] = hashlib.sha256((Path(directory) / "frames.csv").read_bytes()).hexdigest()
    return summary


def reference_summary(directory):
    with (Path(directory) / "reference.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    count = len(rows)
    if count < 100:
        raise ValueError("Reference evaluation needs at least 100 independent trials")
    result = {"trials": count, "first_seed": int(rows[0]["seed"]),
              "sampling": "independent final-epoch samples; no gating or selection"}
    for name, dimensions in (("nis", 2), ("nees", 4)):
        values = [float(row[name]) for row in rows]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("Invalid reference statistic")
        degrees = dimensions * count
        bounds = [dimensions * (1 - 2 / (9 * degrees) + z * math.sqrt(2 / (9 * degrees))) ** 3
                  for z in (-1.959963984540054, 1.959963984540054)]
        mean = statistics.mean(values)
        result[name] = {"mean": mean, "expected_mean": dimensions, "mean_ci95_bounds": bounds,
                        "passed": bounds[0] <= mean <= bounds[1]}
    result["interval_method"] = "Wilson-Hilferty approximation to chi-square ensemble bounds"
    result["passed"] = result["nis"]["passed"] and result["nees"]["passed"]
    return result
