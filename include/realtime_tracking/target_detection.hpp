#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include <cstdint>
#include <optional>
#include <vector>

struct TargetDetection
{
    double center_x;
    double center_y;
    double pixel_area;
};

enum class ThresholdMode
{
    Fixed,
    Otsu
};

struct TargetDetectorConfig
{
    ThresholdMode threshold_mode{ThresholdMode::Fixed};
    std::uint8_t threshold{127U};
    int minimum_pixel_area{1};
    int opening_kernel_size{1};
    std::optional<double> maximum_bounding_box_aspect_ratio{};
    std::optional<double> minimum_circularity{};
    std::optional<cv::Rect> search_region{};
    int smoothing_kernel_size{1};
};

[[nodiscard]] std::vector<TargetDetection> detect_bright_target_candidates(
    const cv::Mat& frame,
    const TargetDetectorConfig& configuration);

[[nodiscard]] std::optional<TargetDetection> detect_bright_target(
    const cv::Mat& frame,
    const TargetDetectorConfig& configuration);

[[nodiscard]] std::optional<TargetDetection> detect_bright_target(
    const cv::Mat& frame,
    std::uint8_t threshold,
    int minimum_pixel_area = 1,
    int opening_kernel_size = 1);
