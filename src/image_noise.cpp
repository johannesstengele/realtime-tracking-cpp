#include "realtime_tracking/image_noise.hpp"
#include "frame_validation.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

cv::Mat add_gaussian_sensor_noise(
    const cv::Mat& clean_frame,
    double standard_deviation,
    std::uint64_t seed)
{
    tracking_detail::validate_grayscale_frame(clean_frame);
    if (!std::isfinite(standard_deviation) || standard_deviation < 0.0)
    {
        throw std::invalid_argument{"Noise standard deviation must be finite and nonnegative."};
    }
    if (standard_deviation == 0.0)
    {
        return clean_frame.clone();
    }

    cv::Mat signed_noise{clean_frame.size(), CV_64FC1};
    cv::RNG generator{seed};
    generator.fill(signed_noise, cv::RNG::NORMAL, 0.0, 1.0);
    const double saturation_limit = 255.0 / standard_deviation;

    for (int row{}; row < signed_noise.rows; ++row)
    {
        double* samples = signed_noise.ptr<double>(row);
        for (int column{}; column < signed_noise.cols; ++column)
        {
            samples[column] = std::clamp(samples[column], -saturation_limit, saturation_limit)
                * standard_deviation;
        }
    }

    cv::Mat noisy_frame;
    cv::add(clean_frame, signed_noise, noisy_frame, cv::noArray(), CV_8U);
    return noisy_frame;
}
