#pragma once

#include <opencv2/core/types.hpp>

[[nodiscard]] cv::Rect make_centered_search_region(
    const cv::Point2d& center,
    const cv::Size& window_size,
    const cv::Size& frame_size);
