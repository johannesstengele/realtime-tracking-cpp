# Realtime Tracking C++

[![Build and evaluate](https://github.com/johannesstengele/realtime-tracking-cpp/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/johannesstengele/realtime-tracking-cpp/actions/workflows/build.yml)

A C++20 library for reproducible motion simulation, uncertainty propagation and image-based tracking experiments.

The complete demonstrator tracks a single simulated image target with prediction-driven search, statistical candidate association and bounded prediction through missing measurements. The evaluation covers seven scenarios, independent calibration and validation seeds, Monte Carlo confidence intervals, Kalman consistency and measured soft real-time latency. See the [requirements and delivery plan](docs/implementation-plan.md) and [technical results](docs/technical-report.md).

The numerical `tracking` library provides trajectory simulation, seeded position noise, position RMSE, CSV export, measurement statistics and linear Kalman prediction and correction. The OpenCV-based `vision` library provides synthetic images, seeded sensor noise, configurable component detection and stateful local tracking with full-frame reacquisition. State vectors use the order `(x, y, vx, vy)`; image coordinates use `x = column`, `y = row`.

## Build and test

Requirements: a C++20 compiler, CMake 3.21 or newer, Git and OpenCV 4 with the `core` and `imgproc` components. CMake uses an installed Eigen 3.4 or fetches Eigen 3.4.0, and fetches GoogleTest 1.18.0 when testing is enabled. The first configuration may need internet access.

Python 3.10 or newer is required for the evaluation tests. With a compiler, CMake, Git and Python available, a fresh checkout can provision pinned dependencies, build and run all tests with:

```sh
python tools/build_project.py --preset dev
python tools/build_project.py --preset release
```

This uses the versioned `vcpkg.json` baseline, minimal OpenCV 4.14.0, Eigen 3.4.0 and GoogleTest 1.18.0. Dependencies and builds stay under the ignored `out/` directory. Existing dependency-manager checkouts are validated and preserved. Use `--vcpkg-root PATH` to reuse a checkout at the manifest baseline. Managed builds use `out/managed/dev` and `out/managed/release`; they are separate from builds using manually installed dependencies. The GitHub Actions workflow exercises both Windows and Linux, plus Linux project sanitizers.

For an existing compatible OpenCV installation, use the original presets:

```sh
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
```

If OpenCV is outside the compiler's search paths, supply its package directory on the first configuration:

```sh
cmake --preset dev -DOpenCV_DIR="/path/to/opencv/build-or-package-directory"
```

The directory must contain `OpenCVConfig.cmake` for the selected compiler and architecture. Machine-specific settings can also go in the ignored `CMakeUserPresets.json`.

On Windows, OpenCV and the application must use matching runtime libraries. This project uses the dynamic MSVC runtime (`/MD`, `/MDd`); when building static OpenCV, configure it with `BUILD_WITH_STATIC_CRT=OFF`. Both Debug and Release OpenCV libraries are needed to test both configurations.

The `release` preset builds and tests optimized code:

```sh
cmake --preset release
cmake --build --preset release
ctest --preset release
```

Each preset has its own build directory under `out/build`. Project warnings are treated as errors in the developer presets; third-party targets retain their own warning settings.

## Runtime diagnostics

```sh
cmake --preset sanitized
cmake --build --preset sanitized
ctest --preset sanitized
```

This uses RelWithDebInfo and AddressSanitizer on MSVC, or AddressSanitizer plus UndefinedBehaviorSanitizer on GCC and Clang. MSVC's Debug runtime checks are incompatible with AddressSanitizer, so use the supplied sanitized build preset. Static C++ dependencies, including OpenCV, must be built with matching sanitizer and runtime settings; pass their package directory when configuring the sanitized tree. GoogleTest is instrumented automatically. The MSVC AddressSanitizer runtime is copied beside project executables.

## Tracking example

After a developer build, run `out/build/dev/Debug/realtime_tracking.exe` with Visual Studio, or `out/build/dev/realtime_tracking` with a single-configuration generator.

The example generates 120 grayscale frames containing a moving disk, adds seeded sensor noise and runs `KalmanTargetTracker`. It reports detection success, detection sources, measurement and filtered position RMSE, and the final velocity estimate. The camera is fixed-depth, frames are spaced at 1/30 second, positions are pixels and velocities are pixels per second. The example produces console output and runs as a CTest integration check.

## Monte Carlo evaluation

After a managed Release build, run on Windows:

```sh
python -m evaluation --executable out/managed/release/Release/tracking_experiment.exe --output out/experiments/validation --strict
```

On Linux, use `out/managed/release/tracking_experiment`. With the original Release preset on Windows, use `out/build/release/Release/tracking_experiment.exe`. Choose a fresh output directory for each experiment; existing results are preserved.

The command calibrates measurement variance on 20 nominal sequences (seeds 100..119), validates seven scenarios on 100 sequences each (seeds 20000..20099) and runs 10,000 independent linear-Gaussian consistency trials (seeds 1000000..1009999). `--strict` fails if the required evidence, accuracy, lifecycle or p99 latency criteria are not met. The Python analysis uses the standard library.

Outputs include per-frame `frames.csv`, complete `configuration.json`, independent reference `reference.csv` and aggregate `report.json`. All accepted measurements, missing estimates, ground truth, state errors, uncertainty diagnostics and stage timings are retained. For direct C++ use:

```sh
tracking_experiment --output out/experiments/manual --scenario all --runs 100 --seed 20000
```

The Python boundary is a versioned CSV/JSON command-line interface; tracking algorithms remain in C++. Analyze existing raw results with `python -m evaluation --input RAW_DIRECTORY --output NEW_REPORT_DIRECTORY`. This analysis-only mode reports scenario results but does not assert complete calibration and reference evidence.

Optional standalone plots use Matplotlib:

```sh
python -m pip install -r evaluation/requirements-plot.txt
python -m evaluation.plot_results out/experiments/validation/report.json --output out/experiments/validation/plots
```

The reference report and plots are in [docs/results](docs/results). Latency includes rendering and tracking, excludes CSV writing and discards the first ten frames of each run for steady-state summaries. A shared desktop or CI host provides measured latency, not a hard real-time bound.

## Detection and tracking

`TargetDetectorConfig` selects a fixed or Otsu threshold, an inclusive minimum pixel area, an optional elliptical opening kernel, and optional aspect-ratio and contour-circularity constraints. A search region must lie entirely inside the frame. Detection coordinates always refer to the full image.

`detect_bright_target_candidates` returns every admissible component; vector order carries no ranking guarantee. `detect_bright_target` retains largest-area selection, including the earlier scalar-parameter overload.

`TargetTracker` searches around its previous detection first, searches the full image after a local miss, and clears its track after a complete miss. It reports initialization, local detection and reacquisition explicitly. Reset clears temporal state. Invalid inputs preserve the existing track. Boundary-adjacent search windows are shifted to retain their dimensions.

`KalmanTargetTracker` integrates a Gaussian state and continuous white-acceleration process model. It searches around the predicted position, chooses the candidate with minimum squared Mahalanobis innovation inside its gate and falls back to a full-frame gated search. It returns `Searching`, `Tracking`, `Coasting` or `Lost`; up to five consecutive misses retain a predicted state. A subsequent initialization receives a new monotonic track identity. Reset preserves identity uniqueness. Invalid frames and time steps preserve state. Per-stage timing is optional and disabled by default.

Detection optionally smooths the image with an odd-sized Gaussian kernel before thresholding. Initial acquisition still chooses the largest admissible component: the operating envelope requires the intended target to satisfy that assumption before clutter appears.

Largest-area association remains a heuristic: plausible clutter inside the local window can keep a track active. Otsu thresholding assumes useful separation in the local intensity histogram. Synthetic tests characterize the stated scenes and noise settings; empirical detector covariance depends on those conditions.

## Prediction benchmark

```sh
cmake --preset benchmark-release
cmake --build --preset benchmark-release
```

Run `out/build/benchmark-release/Release/prediction_benchmark.exe` with a Visual Studio generator, or `out/build/benchmark-release/prediction_benchmark` with a single-configuration generator. It warms up the predictor, then times 100,000 predictions and reports mean latency and a checksum. Run several times on an otherwise idle machine. These are batch averages and do not establish a worst-case latency or real-time guarantee. Benchmarks require sanitizers to be disabled.

## Numerical contracts

Trajectory simulation includes the initial state. Position noise returns an independent trajectory, preserves velocity and takes an explicit seed. Normal-distribution samples can differ between standard-library implementations, so reproducibility is scoped to the same implementation.

Position RMSE requires nonempty sequences of equal length and finite positions. Sequence evaluation reports RMSE over successful detections together with success rate and source counts. CSV export uses a fixed decimal point and round-trip double precision. Measurement statistics estimate bias and unbiased covariance from at least two finite error samples.

Gaussian prediction requires finite means and symmetric, positive-semidefinite covariances, checked with floating-point tolerance. Two process-noise assumptions have separate interfaces:

- `GaussianPredictor(dt, acceleration_sigma)` and the three-argument prediction function use independent constant accelerations over each interval: `Q = sigma_a^2 * G * G^T`.
- `make_constant_velocity_model(dt, acceleration_spectral_density)` returns a `ConstantVelocityProcessModel` for continuous white acceleration, with position variance `q * dt^3 / 3`, position-velocity covariance `q * dt^2 / 2` and velocity variance `q * dt`. The two-argument prediction function accepts this model. The original deterministic `ConstantVelocityModel` remains available.

The position observation selects the first two state coordinates. Innovation covariance combines prior position covariance and measurement covariance. Correction solves with LDLT and uses the Joseph covariance update. Singular innovation covariance is rejected. Position measurements can correct velocity through position-velocity cross-covariances.

The tests combine exact numerical examples, parameterized covariance invariants, input validation and reproducible integration experiments. No test depends on benchmark timing.
