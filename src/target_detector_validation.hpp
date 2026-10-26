#pragma once

#include "realtime_tracking/target_detection.hpp"

#include <cmath>
#include <stdexcept>

namespace tracking_detail
{
inline void validate_target_detector_configuration(const TargetDetectorConfig& configuration)
{
    switch (configuration.threshold_mode)
    {
    case ThresholdMode::Fixed:
    case ThresholdMode::Otsu:
        break;
    default:
        throw std::invalid_argument{"Unsupported threshold mode."};
    }

    if (configuration.minimum_pixel_area <= 0)
    {
        throw std::invalid_argument{"Minimum component area must be positive."};
    }
    if (configuration.opening_kernel_size <= 0 || configuration.opening_kernel_size % 2 == 0)
    {
        throw std::invalid_argument{"Opening kernel size must be positive and odd."};
    }
    if (configuration.smoothing_kernel_size <= 0 || configuration.smoothing_kernel_size % 2 == 0)
    {
        throw std::invalid_argument{"Smoothing kernel size must be positive and odd."};
    }
    if (configuration.maximum_bounding_box_aspect_ratio.has_value()
        && (!std::isfinite(*configuration.maximum_bounding_box_aspect_ratio)
            || *configuration.maximum_bounding_box_aspect_ratio < 1.0))
    {
        throw std::invalid_argument{"Maximum aspect ratio must be finite and at least one."};
    }
    if (configuration.minimum_circularity.has_value()
        && (!std::isfinite(*configuration.minimum_circularity)
            || *configuration.minimum_circularity <= 0.0 || *configuration.minimum_circularity > 1.0))
    {
        throw std::invalid_argument{"Minimum circularity must be finite and in (0, 1]."};
    }
}
}
