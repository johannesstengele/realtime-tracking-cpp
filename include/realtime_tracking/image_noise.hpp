#pragma once

#include <opencv2/core/mat.hpp>

#include <cstdint>

[[nodiscard]] cv::Mat add_gaussian_sensor_noise(
    const cv::Mat& clean_frame,
    double standard_deviation,
    std::uint64_t seed);
