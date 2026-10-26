#pragma once

#include <opencv2/core/mat.hpp>

#include <stdexcept>

namespace tracking_detail
{
inline void validate_grayscale_frame(const cv::Mat& frame)
{
    if (frame.empty() || frame.dims != 2 || frame.type() != CV_8UC1)
    {
        throw std::invalid_argument{"Frame must be a nonempty, two-dimensional CV_8UC1 image."};
    }
}
}
