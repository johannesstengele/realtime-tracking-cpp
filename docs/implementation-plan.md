# Tracking evaluation delivery plan

The deliverable is a single-target, image-based C++ tracking demonstrator with reproducible Python evaluation. The operating envelope is simulated grayscale imagery, a fixed-depth pinhole camera, one bright target, bounded clutter and explicitly scripted visibility interruptions. Positions are pixels, velocities are pixels per second and time is seconds.

## Delivery order

1. Define the operating envelope and measurable acceptance criteria.
2. Integrate prediction, candidate association, search and track lifecycle in a reusable Kalman tracker.
3. Add deterministic camera, motion, image-noise, clutter and dropout scenarios.
4. Add a machine-readable experiment runner and Python Monte Carlo analysis with uncertainty estimates.
5. Measure per-frame pipeline and stage latency against the frame budget.
6. Add dependency provisioning, continuous integration and a reproducible technical results report.

This repository starts from a published source snapshot. Changes are validated with full developer, release and sanitizer build and test runs. Existing APIs remain available.

## Acceptance criteria

| Requirement | Acceptance |
| --- | --- |
| Operating envelope | 320 x 240, 30 frames/s, 120 frames/run, fixed-depth camera; exact scenario definitions exported with results |
| Nominal tracking | At least 99% correctly associated visible frames after initialization; position RMSE <= 1 pixel |
| Maneuver tracking | At least 95% correctly associated visible frames; position RMSE <= 3 pixels |
| Clutter and degraded images | At least 95% correct visible-frame associations; wrong associations <= 1%; position RMSE <= 3 pixels |
| Short interruption | Preserve track identity through three missing frames and accept the returning target within two frames |
| Long interruption | Drop a track after the configured coast limit and initialize a new identity within two visible frames |
| Statistical evaluation | At least 100 independent seeds per scenario, disjoint calibration/validation seeds, run-level bootstrap 95% intervals; report failures and coverage alongside conditional errors |
| Uncertainty | Report position covariance coverage and accepted-innovation statistics; separately verify Kalman consistency on an unselected linear-Gaussian reference experiment |
| Runtime | Release end-to-end processing p99 <= 33.333 ms on the documented reference machine; report observed maximum, jitter and deadline misses, without claiming a hard real-time bound |
| Reproducibility | Document dependency versions, build commands, experiment configuration and machine details; CI builds/tests on Windows and Linux |
| Delivery | C++ runner, Python standard-library analysis, CSV/JSON results, standalone plots and technical report |

Correct association means an accepted target measurement within 3 pixels of ground truth. Missing estimates, rejected measurements and invisible frames are recorded separately. Error metrics on successful associations are never presented without association and estimate-availability rates. The initial target must be the largest admissible component; identity after loss cannot be inferred from appearance alone.

## Boundaries

This is a software simulation and measured soft real-time demonstrator. Multi-target tracking, learned detection, perspective depth estimation, real-camera validation, hardware-in-the-loop and formal worst-case execution-time analysis are future extensions. Detector covariance is scenario dependent; validation results must not be interpreted as universal camera calibration.

## Delivered evidence

All six implementation parts are complete. `KalmanTargetTracker` integrates search, association and lifecycle; `tracking_experiment` renders and exports seven deterministic scenarios; `python -m evaluation` performs calibration, Monte Carlo analysis and independent Kalman consistency checks. `tools/build_project.py` provisions the pinned dependency manifest and runs the build and test suite.

The [technical report](technical-report.md) records 700 validation sequences, 84,000 image frames, 10,000 reference trials and the measured latency distributions. The [Windows and Linux workflow](https://github.com/johannesstengele/realtime-tracking-cpp/actions/workflows/build.yml) independently verifies Debug, Release, evaluation and Linux sanitizer checks. Local Windows diagnostics also use instrumented OpenCV. The [source manifest](results/source-manifest.json) identifies the tested source files independently of Git history.
