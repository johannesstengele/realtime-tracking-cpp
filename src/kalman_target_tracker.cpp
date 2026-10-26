#include "realtime_tracking/kalman_target_tracker.hpp"

#include "realtime_tracking/position_measurement.hpp"
#include "realtime_tracking/search_region.hpp"
#include "covariance_validation.hpp"
#include "frame_validation.hpp"
#include "target_detector_validation.hpp"

#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <utility>

namespace
{
template <typename Function>
auto measure(bool enabled, double& nanoseconds, Function&& function)
{
    if (!enabled)
    {
        return function();
    }
    const auto start = std::chrono::steady_clock::now();
    auto value = function();
    nanoseconds += std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count();
    return value;
}

struct Association
{
    TargetDetection detection;
    double squared_distance;
};

std::optional<Association> associate(const std::vector<TargetDetection>& candidates,
    const GaussianState& prediction, const KalmanTrackerConfig& configuration)
{
    std::optional<Association> selected;
    for (const auto& candidate : candidates)
    {
        const auto innovation = compute_position_innovation(prediction,
            {candidate.center_x, candidate.center_y}, configuration.measurement_covariance);
        const Eigen::LDLT<Eigen::Matrix2d> factorization{innovation.innovation_covariance};
        if (factorization.info() != Eigen::Success || factorization.vectorD().minCoeff() <= 0.0)
        {
            throw std::runtime_error{"Association covariance must be positive definite."};
        }
        const double distance = innovation.innovation.dot(factorization.solve(innovation.innovation));
        if (std::isfinite(distance) && distance >= 0.0
            && distance <= configuration.squared_gate_threshold
            && (!selected.has_value() || distance < selected->squared_distance))
        {
            selected = Association{candidate, distance};
        }
    }
    return selected;
}
}

KalmanTargetTracker::KalmanTargetTracker(KalmanTrackerConfig configuration)
    : configuration_{std::move(configuration)}
{
    tracking_detail::validate_target_detector_configuration(configuration_.detector);
    tracking_detail::validate_covariance(configuration_.measurement_covariance);
    const Eigen::LDLT<Eigen::Matrix2d> factorization{configuration_.measurement_covariance};
    if (configuration_.detector.search_region.has_value()
        || configuration_.search_window.width <= 0 || configuration_.search_window.height <= 0
        || configuration_.search_window.width % 2 == 0 || configuration_.search_window.height % 2 == 0
        || !std::isfinite(configuration_.initial_velocity_variance)
        || configuration_.initial_velocity_variance <= 0.0
        || !std::isfinite(configuration_.squared_gate_threshold)
        || configuration_.squared_gate_threshold <= 0.0
        || factorization.info() != Eigen::Success || factorization.vectorD().minCoeff() <= 0.0)
    {
        throw std::invalid_argument{"Invalid Kalman tracker configuration."};
    }
    static_cast<void>(make_constant_velocity_model(1.0, configuration_.acceleration_spectral_density));
}

KalmanTrackingUpdate KalmanTargetTracker::update(const cv::Mat& frame, double delta_time)
{
    const auto start = configuration_.collect_timings ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    tracking_detail::validate_grayscale_frame(frame);
    if (configuration_.search_window.width > frame.cols || configuration_.search_window.height > frame.rows)
    {
        throw std::invalid_argument{"Search window must fit inside the frame."};
    }
    const auto model = make_constant_velocity_model(delta_time, configuration_.acceleration_spectral_density);
    KalmanTrackingUpdate result{};
    result.track_id = last_track_id_;
    if (!posterior_.has_value())
    {
        const auto detection = measure(configuration_.collect_timings, result.timings.detection_ns,
            [&] { return detect_bright_target(frame, configuration_.detector); });
        if (!detection.has_value())
        {
            if (configuration_.collect_timings)
            {
                result.timings.total_ns = std::chrono::duration<double, std::nano>(
                    std::chrono::steady_clock::now() - start).count();
            }
            return result;
        }
        if (last_track_id_ == std::numeric_limits<std::uint64_t>::max())
        {
            throw std::overflow_error{"Track identity range exhausted."};
        }
        GaussianState initialized{Eigen::Vector4d{detection->center_x, detection->center_y, 0.0, 0.0},
            Eigen::Matrix4d::Zero()};
        initialized.covariance.topLeftCorner<2, 2>() = configuration_.measurement_covariance;
        initialized.covariance.bottomRightCorner<2, 2>() =
            Eigen::Matrix2d::Identity() * configuration_.initial_velocity_variance;
        result.status = TrackStatus::Tracking;
        result.state = initialized;
        result.detection = detection;
        result.source = DetectionSource::InitialFullFrame;
        result.track_id = last_track_id_ + 1U;
    }
    else
    {
        const auto prediction = measure(configuration_.collect_timings, result.timings.prediction_ns,
            [&] { return predict_gaussian_state(*posterior_, model); });
        std::optional<Association> selected;
        DetectionSource source{DetectionSource::Reacquisition};
        if (prediction.mean(0) >= 0.0 && prediction.mean(0) < static_cast<double>(frame.cols)
            && prediction.mean(1) >= 0.0 && prediction.mean(1) < static_cast<double>(frame.rows))
        {
            auto local = configuration_.detector;
            local.search_region = make_centered_search_region(
                {prediction.mean(0), prediction.mean(1)}, configuration_.search_window, frame.size());
            const auto candidates = measure(configuration_.collect_timings, result.timings.detection_ns,
                [&] { return detect_bright_target_candidates(frame, local); });
            selected = measure(configuration_.collect_timings, result.timings.association_ns,
                [&] { return associate(candidates, prediction, configuration_); });
            source = DetectionSource::LocalSearch;
        }
        if (!selected.has_value())
        {
            const auto candidates = measure(configuration_.collect_timings, result.timings.detection_ns,
                [&] { return detect_bright_target_candidates(frame, configuration_.detector); });
            selected = measure(configuration_.collect_timings, result.timings.association_ns,
                [&] { return associate(candidates, prediction, configuration_); });
            source = DetectionSource::Reacquisition;
        }
        if (selected.has_value())
        {
            result.state = measure(configuration_.collect_timings, result.timings.correction_ns,
                [&] { return correct_position_measurement(prediction,
                    {selected->detection.center_x, selected->detection.center_y}, configuration_.measurement_covariance); });
            result.detection = selected->detection;
            result.source = source;
            result.normalized_innovation_squared = selected->squared_distance;
            result.status = TrackStatus::Tracking;
        }
        else if (consecutive_misses_ < configuration_.maximum_coast_frames)
        {
            result.state = prediction;
            result.consecutive_misses = consecutive_misses_ + 1U;
            result.status = TrackStatus::Coasting;
        }
        else
        {
            result.status = TrackStatus::Lost;
        }
    }
    posterior_ = result.state;
    consecutive_misses_ = result.consecutive_misses;
    last_track_id_ = result.track_id;
    if (configuration_.collect_timings)
    {
        result.timings.total_ns = std::chrono::duration<double, std::nano>(
            std::chrono::steady_clock::now() - start).count();
    }
    return result;
}

void KalmanTargetTracker::reset() noexcept
{
    posterior_.reset();
    consecutive_misses_ = 0U;
}

bool KalmanTargetTracker::has_track() const noexcept
{
    return posterior_.has_value();
}
