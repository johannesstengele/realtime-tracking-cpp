#include "realtime_tracking/target_tracker.hpp"

#include "realtime_tracking/search_region.hpp"
#include "frame_validation.hpp"
#include "target_detector_validation.hpp"

#include <stdexcept>
#include <utility>

TargetTracker::TargetTracker(TargetDetectorConfig detector_configuration, cv::Size local_search_window)
    : detector_configuration_{std::move(detector_configuration)},
      local_search_window_{local_search_window}
{
    tracking_detail::validate_target_detector_configuration(detector_configuration_);
    if (detector_configuration_.search_region.has_value())
    {
        throw std::invalid_argument{"The tracker manages the detector search region."};
    }
    if (local_search_window_.width <= 0 || local_search_window_.height <= 0
        || local_search_window_.width % 2 == 0 || local_search_window_.height % 2 == 0)
    {
        throw std::invalid_argument{"Local search-window dimensions must be positive and odd."};
    }
}

std::optional<TrackingUpdate> TargetTracker::update(const cv::Mat& frame)
{
    tracking_detail::validate_grayscale_frame(frame);
    if (local_search_window_.width > frame.cols || local_search_window_.height > frame.rows)
    {
        throw std::invalid_argument{"Local search window must fit inside the frame."};
    }

    const bool had_track = previous_detection_.has_value();
    if (had_track && previous_detection_->center_x < static_cast<double>(frame.cols)
        && previous_detection_->center_y < static_cast<double>(frame.rows))
    {
        auto local_configuration = detector_configuration_;
        local_configuration.search_region = make_centered_search_region(
            {previous_detection_->center_x, previous_detection_->center_y},
            local_search_window_, frame.size());
        const auto local_detection = detect_bright_target(frame, local_configuration);
        if (local_detection.has_value())
        {
            previous_detection_ = *local_detection;
            return TrackingUpdate{*local_detection, DetectionSource::LocalSearch};
        }
    }

    const auto full_frame_detection = detect_bright_target(frame, detector_configuration_);
    if (!full_frame_detection.has_value())
    {
        previous_detection_.reset();
        return std::nullopt;
    }

    previous_detection_ = *full_frame_detection;
    return TrackingUpdate{*full_frame_detection,
        had_track ? DetectionSource::Reacquisition : DetectionSource::InitialFullFrame};
}

void TargetTracker::reset() noexcept
{
    previous_detection_.reset();
}

bool TargetTracker::has_track() const noexcept
{
    return previous_detection_.has_value();
}
