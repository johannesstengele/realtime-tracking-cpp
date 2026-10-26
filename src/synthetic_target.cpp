#include "realtime_tracking/synthetic_target.hpp"

#include <opencv2/imgproc.hpp>

#include <stdexcept>

cv::Mat make_synthetic_target_frame(const SyntheticTargetSpec& specification)
{
    if (specification.width <= 0 || specification.height <= 0)
    {
        throw std::invalid_argument{"Image dimensions must be positive."};
    }

    if (specification.center_x < 0 || specification.center_x >= specification.width
        || specification.center_y < 0 || specification.center_y >= specification.height)
    {
        throw std::invalid_argument{"Target center must lie inside the image."};
    }

    if (specification.radius <= 0)
    {
        throw std::invalid_argument{"Target radius must be positive."};
    }

    if (specification.radius > specification.center_x
        || specification.radius >= specification.width - specification.center_x
        || specification.radius > specification.center_y
        || specification.radius >= specification.height - specification.center_y)
    {
        throw std::invalid_argument{"Complete target disk must lie inside the image."};
    }

    cv::Mat frame{specification.height, specification.width, CV_8UC1,
                  cv::Scalar{static_cast<double>(specification.background_intensity)}};
    cv::circle(frame, cv::Point{specification.center_x, specification.center_y},
        specification.radius, cv::Scalar{static_cast<double>(specification.target_intensity)},
        cv::FILLED, cv::LINE_8);
    return frame;
}
