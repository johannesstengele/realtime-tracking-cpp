#include "realtime_tracking/target_detection.hpp"
#include "frame_validation.hpp"
#include "target_detector_validation.hpp"

#include <opencv2/imgproc.hpp>

#include <cstddef>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <numbers>
#include <vector>
#include <utility>

namespace
{
double component_circularity(const cv::Mat& labels, int label, const cv::Rect& bounds)
{
    const cv::Mat mask = labels(bounds) == label;
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    double area{};
    double perimeter{};

    for (const auto& contour : contours)
    {
        area += cv::contourArea(contour);
        perimeter += cv::arcLength(contour, true);
    }

    return perimeter > 0.0 ? 4.0 * std::numbers::pi * area / (perimeter * perimeter) : 0.0;
}
}

std::vector<TargetDetection> detect_bright_target_candidates(
    const cv::Mat& frame,
    const TargetDetectorConfig& configuration)
{
    tracking_detail::validate_grayscale_frame(frame);
    tracking_detail::validate_target_detector_configuration(configuration);

    const cv::Rect search_region = configuration.search_region.value_or(
        cv::Rect{0, 0, frame.cols, frame.rows});
    if (search_region.x < 0 || search_region.y < 0
        || search_region.width <= 0 || search_region.height <= 0
        || search_region.x >= frame.cols || search_region.y >= frame.rows
        || search_region.width > frame.cols - search_region.x
        || search_region.height > frame.rows - search_region.y)
    {
        throw std::invalid_argument{"Search region must be nonempty and entirely inside the frame."};
    }

    cv::Mat analysis_frame = frame(search_region);
    if (analysis_frame.total() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error{"Image exceeds the connected-component area range."};
    }

    const bool use_otsu = configuration.threshold_mode == ThresholdMode::Otsu;
    const int threshold_type = cv::THRESH_BINARY | (use_otsu ? cv::THRESH_OTSU : 0);
    const double threshold_value = use_otsu ? 0.0 : static_cast<double>(configuration.threshold);
    if (configuration.smoothing_kernel_size > 1)
    {
        cv::Mat smoothed;
        cv::GaussianBlur(analysis_frame, smoothed,
            {configuration.smoothing_kernel_size, configuration.smoothing_kernel_size}, 0.0);
        analysis_frame = std::move(smoothed);
    }
    cv::Mat binary_mask;
    static_cast<void>(cv::threshold(analysis_frame, binary_mask, threshold_value, 255.0, threshold_type));

    if (configuration.opening_kernel_size > 1)
    {
        const auto kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
            cv::Size{configuration.opening_kernel_size, configuration.opening_kernel_size});
        cv::morphologyEx(binary_mask, binary_mask, cv::MORPH_OPEN, kernel);
    }

    cv::Mat labels;
    cv::Mat statistics;
    cv::Mat centroids;
    const int label_count = cv::connectedComponentsWithStats(
        binary_mask, labels, statistics, centroids, 8, CV_32S);
    std::vector<TargetDetection> candidates;
    candidates.reserve(static_cast<std::size_t>(label_count - 1));

    for (int label{1}; label < label_count; ++label)
    {
        const int area = statistics.at<int>(label, cv::CC_STAT_AREA);
        const int width = statistics.at<int>(label, cv::CC_STAT_WIDTH);
        const int height = statistics.at<int>(label, cv::CC_STAT_HEIGHT);
        const double aspect_ratio = static_cast<double>(std::max(width, height))
            / static_cast<double>(std::min(width, height));
        const bool shape_is_admissible = !configuration.maximum_bounding_box_aspect_ratio.has_value()
            || aspect_ratio <= *configuration.maximum_bounding_box_aspect_ratio;

        if (area < configuration.minimum_pixel_area || !shape_is_admissible)
        {
            continue;
        }

        if (configuration.minimum_circularity.has_value())
        {
            const cv::Rect bounds{statistics.at<int>(label, cv::CC_STAT_LEFT),
                statistics.at<int>(label, cv::CC_STAT_TOP), width, height};
            if (component_circularity(labels, label, bounds) < *configuration.minimum_circularity)
            {
                continue;
            }
        }

        candidates.push_back(TargetDetection{
            centroids.at<double>(label, 0) + static_cast<double>(search_region.x),
            centroids.at<double>(label, 1) + static_cast<double>(search_region.y),
            static_cast<double>(area)});
    }

    return candidates;
}

std::optional<TargetDetection> detect_bright_target(
    const cv::Mat& frame,
    const TargetDetectorConfig& configuration)
{
    const auto candidates = detect_bright_target_candidates(frame, configuration);
    if (candidates.empty())
    {
        return std::nullopt;
    }

    return *std::max_element(candidates.cbegin(), candidates.cend(),
        [](const TargetDetection& left, const TargetDetection& right)
        {
            return left.pixel_area < right.pixel_area;
        });
}

std::optional<TargetDetection> detect_bright_target(
    const cv::Mat& frame,
    std::uint8_t threshold,
    int minimum_pixel_area,
    int opening_kernel_size)
{
    TargetDetectorConfig configuration{};
    configuration.threshold = threshold;
    configuration.minimum_pixel_area = minimum_pixel_area;
    configuration.opening_kernel_size = opening_kernel_size;
    return detect_bright_target(frame, configuration);
}
