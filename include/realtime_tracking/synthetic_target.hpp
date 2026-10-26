#pragma once

#include <opencv2/core/mat.hpp>

#include <cstdint>

struct SyntheticTargetSpec
{
    int width;
    int height;
    int center_x;
    int center_y;
    int radius;
    std::uint8_t background_intensity;
    std::uint8_t target_intensity;
};

[[nodiscard]] cv::Mat make_synthetic_target_frame(const SyntheticTargetSpec& specification);
