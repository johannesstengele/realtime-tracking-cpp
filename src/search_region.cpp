#include "realtime_tracking/search_region.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

cv::Rect make_centered_search_region(
    const cv::Point2d& center,
    const cv::Size& window_size,
    const cv::Size& frame_size)
{
    if (!std::isfinite(center.x) || !std::isfinite(center.y))
    {
        throw std::invalid_argument{"Search-region center must be finite."};
    }
    if (frame_size.width <= 0 || frame_size.height <= 0)
    {
        throw std::invalid_argument{"Frame dimensions must be positive."};
    }
    if (window_size.width <= 0 || window_size.height <= 0
        || window_size.width % 2 == 0 || window_size.height % 2 == 0)
    {
        throw std::invalid_argument{"Window dimensions must be positive and odd."};
    }
    if (window_size.width > frame_size.width || window_size.height > frame_size.height)
    {
        throw std::invalid_argument{"Search window must fit inside the frame."};
    }
    if (center.x < 0.0 || center.y < 0.0
        || center.x >= static_cast<double>(frame_size.width)
        || center.y >= static_cast<double>(frame_size.height))
    {
        throw std::invalid_argument{"Search-region center must lie inside the frame."};
    }

    const int desired_x = static_cast<int>(std::lround(center.x)) - window_size.width / 2;
    const int desired_y = static_cast<int>(std::lround(center.y)) - window_size.height / 2;
    return cv::Rect{std::clamp(desired_x, 0, frame_size.width - window_size.width),
        std::clamp(desired_y, 0, frame_size.height - window_size.height),
        window_size.width, window_size.height};
}
