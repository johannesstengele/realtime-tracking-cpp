#pragma once

#include "realtime_tracking/target_detection.hpp"

#include <opencv2/core/types.hpp>

#include <optional>

enum class DetectionSource
{
    InitialFullFrame,
    LocalSearch,
    Reacquisition
};

struct TrackingUpdate
{
    TargetDetection detection;
    DetectionSource source;
};

class TargetTracker
{
public:
    TargetTracker(TargetDetectorConfig detector_configuration, cv::Size local_search_window);

    [[nodiscard]] std::optional<TrackingUpdate> update(const cv::Mat& frame);
    void reset() noexcept;
    [[nodiscard]] bool has_track() const noexcept;

private:
    TargetDetectorConfig detector_configuration_;
    cv::Size local_search_window_;
    std::optional<TargetDetection> previous_detection_;
};
