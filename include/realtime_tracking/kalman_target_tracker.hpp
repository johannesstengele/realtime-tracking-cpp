#pragma once

#include "realtime_tracking/gaussian_state.hpp"
#include "realtime_tracking/target_tracker.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>

enum class TrackStatus
{
    Searching,
    Tracking,
    Coasting,
    Lost
};

struct KalmanTrackerConfig
{
    TargetDetectorConfig detector{};
    cv::Size search_window{63, 63};
    Eigen::Matrix2d measurement_covariance{Eigen::Matrix2d::Identity() * 0.36};
    double acceleration_spectral_density{64.0};
    double initial_velocity_variance{400.0};
    double squared_gate_threshold{9.210340371976184};
    std::size_t maximum_coast_frames{5};
    bool collect_timings{false};
};

struct TrackingTimings
{
    double prediction_ns{};
    double detection_ns{};
    double association_ns{};
    double correction_ns{};
    double total_ns{};
};

struct KalmanTrackingUpdate
{
    TrackStatus status{TrackStatus::Searching};
    std::optional<GaussianState> state{};
    std::optional<TargetDetection> detection{};
    std::optional<DetectionSource> source{};
    std::optional<double> normalized_innovation_squared{};
    std::uint64_t track_id{};
    std::size_t consecutive_misses{};
    TrackingTimings timings{};
};

class KalmanTargetTracker
{
public:
    explicit KalmanTargetTracker(KalmanTrackerConfig configuration = {});

    [[nodiscard]] KalmanTrackingUpdate update(const cv::Mat& frame, double delta_time);
    void reset() noexcept;
    [[nodiscard]] bool has_track() const noexcept;

private:
    KalmanTrackerConfig configuration_;
    std::optional<GaussianState> posterior_{};
    std::size_t consecutive_misses_{};
    std::uint64_t last_track_id_{};
};
