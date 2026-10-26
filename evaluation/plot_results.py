import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot tracking accuracy and latency with bootstrap intervals")
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = json.loads(arguments.report.read_text(encoding="utf-8"))
    arguments.output.mkdir(parents=True, exist_ok=True)
    names = list(report["scenarios"])
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for index, metric in enumerate(("measurement_rmse", "position_rmse")):
        values = [report["scenarios"][name]["metrics"][metric] for name in names]
        means = [value["mean"] for value in values]
        errors = [[max(0, value["mean"] - value["ci95"][0]) for value in values],
                  [max(0, value["ci95"][1] - value["mean"]) for value in values]]
        axes[0].errorbar([position + index * 0.12 for position in range(len(names))], means,
                         yerr=errors, marker="o", capsize=3, linestyle="none",
                         label="Raw detector (correct hits)" if index == 0 else "Kalman (all available estimates)")
    axes[0].set_ylabel("Position RMSE [pixels]")
    axes[0].set_title("Mean run RMSE with 95% bootstrap intervals")
    axes[0].legend(fontsize=8)
    for statistic in ("median", "p99", "maximum"):
        axes[1].plot(range(len(names)), [report["scenarios"][name]["timings"]["pipeline_ms"][statistic]
                                       for name in names], marker="o", label=statistic)
    axes[1].axhline(report["latency_budget_ms"], linestyle="--", color="black", label="Frame budget")
    axes[1].set_ylabel("End-to-end latency [ms]")
    axes[1].set_title("Rendering plus tracking; CSV writing excluded")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.set_xticks(range(len(names)), [name.replace("_", "\n") for name in names], fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(arguments.output / "tracking-results.svg", metadata={"Date": None})
    figure.savefig(arguments.output / "tracking-results.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
