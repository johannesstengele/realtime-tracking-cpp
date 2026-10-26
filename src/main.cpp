#include "realtime_tracking/kalman_target_tracker.hpp"
#include "realtime_tracking/state2d.hpp"
#include "realtime_tracking/tracking_scenario.hpp"
#include "realtime_tracking/trajectory_analysis.hpp"

#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>

namespace
{
int run_tracking_example()
{
    constexpr int frame_count{120};
    KalmanTrackerConfig configuration{};
    configuration.detector.minimum_pixel_area = 20;
    configuration.detector.opening_kernel_size = 3;
    configuration.detector.smoothing_kernel_size = 3;
    KalmanTargetTracker tracker{configuration};
    const ScenarioConfig simulation{TrackingScenario::Nominal, 1U, frame_count, 1.0 / 30.0};
    std::optional<GaussianState> posterior;
    std::vector<State2D> reference;
    std::vector<State2D> measurements;
    std::vector<State2D> estimates;
    reference.reserve(frame_count);
    measurements.reserve(frame_count);
    estimates.reserve(frame_count);
    int initial_detections{};
    int local_detections{};
    int reacquisitions{};
    int misses{};

    for (int index{}; index < frame_count; ++index)
    {
        const auto frame = render_tracking_frame(simulation, index);
        const auto update = tracker.update(frame.image, simulation.delta_time);
        posterior = update.state;
        if (!update.detection.has_value())
        {
            ++misses;
            continue;
        }

        const auto& detection = *update.detection;
        reference.push_back(frame.truth);
        measurements.push_back({detection.center_x, detection.center_y, 0.0, 0.0});
        estimates.push_back({posterior->mean(0), posterior->mean(1), posterior->mean(2), posterior->mean(3)});
        switch (*update.source)
        {
        case DetectionSource::InitialFullFrame:
            ++initial_detections;
            break;
        case DetectionSource::LocalSearch:
            ++local_detections;
            break;
        case DetectionSource::Reacquisition:
            ++reacquisitions;
            break;
        }
    }

    if (reference.empty() || !posterior.has_value())
    {
        throw std::runtime_error{"No target was detected in the image sequence."};
    }

    std::cout << std::fixed << std::setprecision(6)
              << "frames=" << frame_count << '\n'
              << "detections=" << measurements.size() << '\n'
              << "success_rate=" << static_cast<double>(measurements.size()) / frame_count << '\n'
              << "initial_detections=" << initial_detections << '\n'
              << "local_detections=" << local_detections << '\n'
              << "reacquisitions=" << reacquisitions << '\n'
              << "misses=" << misses << '\n'
              << "measurement_position_rmse=" << position_rmse(reference, measurements) << '\n'
              << "filtered_position_rmse=" << position_rmse(reference, estimates) << '\n'
              << "estimated_velocity_x=" << posterior->mean(2) << '\n'
              << "estimated_velocity_y=" << posterior->mean(3) << '\n';
    return 0;
}
}

int main()
{
    try
    {
        return run_tracking_example();
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
