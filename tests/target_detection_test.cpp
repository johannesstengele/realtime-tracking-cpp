#include "realtime_tracking/synthetic_target.hpp"
#include "realtime_tracking/target_detection.hpp"

#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdint>
#include <stdexcept>
#include <limits>
#include <vector>
#include <algorithm>
#include <cmath>

TEST(TargetDetection, RecoversCenterAndPixelAreaWithoutModifyingInput)
{
    const auto frame = make_synthetic_target_frame({64, 48, 30, 20, 5, 10U, 240U});
    const auto original = frame.clone();
    const auto detection = detect_bright_target(frame, 100U);
    ASSERT_TRUE(detection.has_value());
    EXPECT_NEAR(detection->center_x, 30.0, 1e-12);
    EXPECT_NEAR(detection->center_y, 20.0, 1e-12);
    EXPECT_DOUBLE_EQ(detection->pixel_area, static_cast<double>(cv::countNonZero(frame > 100)));
    EXPECT_EQ(cv::countNonZero(frame != original), 0);
}

TEST(TargetDetection, ThresholdComparisonIsStrict)
{
    cv::Mat frame{2, 3, CV_8UC1, cv::Scalar{100}};
    EXPECT_FALSE(detect_bright_target(frame, 100U).has_value());
    frame.at<std::uint8_t>(1, 2) = 101U;
    const auto detection = detect_bright_target(frame, 100U);
    ASSERT_TRUE(detection.has_value());
    EXPECT_DOUBLE_EQ(detection->center_x, 2.0);
    EXPECT_DOUBLE_EQ(detection->center_y, 1.0);
    EXPECT_DOUBLE_EQ(detection->pixel_area, 1.0);
}

TEST(TargetDetection, RejectsEmptyColorAndNonplanarInput)
{
    const cv::Mat color{16, 16, CV_8UC3, cv::Scalar{10, 10, 10}};
    const int sizes[]{2, 3, 4};
    const cv::Mat volume{3, sizes, CV_8UC1, cv::Scalar{10}};
    EXPECT_THROW(static_cast<void>(detect_bright_target({}, 100U)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(detect_bright_target(color, 100U)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(detect_bright_target(volume, 100U)), std::invalid_argument);
}

TEST(TargetDetection, IgnoresIsolatedBrightClutter)
{
    auto frame = make_synthetic_target_frame({64, 48, 30, 20, 5, 10U, 240U});
    const auto clean = detect_bright_target(frame, 100U, 20);
    frame.at<std::uint8_t>(0, 0) = 240U;
    const auto cluttered = detect_bright_target(frame, 100U, 20);
    ASSERT_TRUE(clean.has_value());
    ASSERT_TRUE(cluttered.has_value());
    EXPECT_NEAR(cluttered->center_x, 30.0, 1e-12);
    EXPECT_NEAR(cluttered->center_y, 20.0, 1e-12);
    EXPECT_DOUBLE_EQ(cluttered->pixel_area, clean->pixel_area);
}

TEST(TargetDetection, UsesEightConnectivityAndInclusiveMinimumArea)
{
    cv::Mat frame{5, 5, CV_8UC1, cv::Scalar{0}};
    frame.at<std::uint8_t>(1, 1) = 255U;
    frame.at<std::uint8_t>(2, 2) = 255U;
    const auto detection = detect_bright_target(frame, 100U, 2);
    ASSERT_TRUE(detection.has_value());
    EXPECT_DOUBLE_EQ(detection->pixel_area, 2.0);
    EXPECT_DOUBLE_EQ(detection->center_x, 1.5);
    EXPECT_DOUBLE_EQ(detection->center_y, 1.5);
    EXPECT_FALSE(detect_bright_target(frame, 100U, 3).has_value());
    EXPECT_THROW(static_cast<void>(detect_bright_target(frame, 100U, 0)), std::invalid_argument);
}

TEST(TargetDetection, OpeningRemovesThinLargerDistractor)
{
    auto frame = make_synthetic_target_frame({64, 48, 30, 20, 2, 10U, 240U});
    for (int column{2}; column <= 21; ++column)
    {
        frame.at<std::uint8_t>(5, column) = 240U;
    }
    const auto original = frame.clone();
    const auto unfiltered = detect_bright_target(frame, 100U, 1, 1);
    const auto opened = detect_bright_target(frame, 100U, 1, 3);
    ASSERT_TRUE(unfiltered.has_value());
    ASSERT_TRUE(opened.has_value());
    EXPECT_NEAR(unfiltered->center_x, 11.5, 1e-12);
    EXPECT_NEAR(unfiltered->center_y, 5.0, 1e-12);
    EXPECT_NEAR(opened->center_x, 30.0, 1e-12);
    EXPECT_NEAR(opened->center_y, 20.0, 1e-12);
    EXPECT_EQ(cv::countNonZero(frame != original), 0);
}

TEST(TargetDetection, RejectsInvalidOpeningKernel)
{
    const cv::Mat frame{16, 16, CV_8UC1, cv::Scalar{0}};
    for (const int kernel : {0, -1, 2})
    {
        EXPECT_THROW(static_cast<void>(detect_bright_target(frame, 100U, 1, kernel)),
            std::invalid_argument);
    }
}

TEST(TargetDetection, AspectRatioRejectsLargerThickBar)
{
    auto frame = make_synthetic_target_frame({80, 60, 50, 30, 5, 10U, 240U});
    cv::rectangle(frame, cv::Point{2, 4}, cv::Point{31, 8}, cv::Scalar{240}, cv::FILLED);
    TargetDetectorConfig configuration{};
    configuration.threshold = 100U;
    configuration.minimum_pixel_area = 20;
    const auto unfiltered = detect_bright_target(frame, configuration);
    ASSERT_TRUE(unfiltered.has_value());
    EXPECT_NEAR(unfiltered->center_x, 16.5, 1e-12);
    EXPECT_NEAR(unfiltered->center_y, 6.0, 1e-12);
    configuration.maximum_bounding_box_aspect_ratio = 2.0;
    const auto filtered = detect_bright_target(frame, configuration);
    ASSERT_TRUE(filtered.has_value());
    EXPECT_NEAR(filtered->center_x, 50.0, 1e-12);
    EXPECT_NEAR(filtered->center_y, 30.0, 1e-12);
}

TEST(TargetDetection, AspectRatioBoundIsInclusiveAndValidated)
{
    cv::Mat frame{8, 8, CV_8UC1, cv::Scalar{0}};
    cv::rectangle(frame, cv::Rect{1, 1, 4, 2}, cv::Scalar{255}, cv::FILLED);
    TargetDetectorConfig configuration{};
    configuration.maximum_bounding_box_aspect_ratio = 2.0;
    EXPECT_TRUE(detect_bright_target(frame, configuration).has_value());
    configuration.maximum_bounding_box_aspect_ratio = 1.99;
    EXPECT_FALSE(detect_bright_target(frame, configuration).has_value());
    for (const double invalid : {0.9, std::numeric_limits<double>::infinity(),
                                 std::numeric_limits<double>::quiet_NaN()})
    {
        configuration.maximum_bounding_box_aspect_ratio = invalid;
        EXPECT_THROW(static_cast<void>(detect_bright_target(frame, configuration)), std::invalid_argument);
    }
}

TEST(TargetDetection, CircularityRejectsRotatedElongatedClutter)
{
    auto frame = make_synthetic_target_frame({100, 80, 70, 55, 5, 10U, 240U});
    const std::vector<cv::Point> rotated_bar{{5, 8}, {9, 4}, {39, 34}, {35, 38}};
    cv::fillConvexPoly(frame, rotated_bar, cv::Scalar{240}, cv::LINE_8);
    TargetDetectorConfig configuration{};
    configuration.threshold = 100U;
    configuration.minimum_pixel_area = 20;
    configuration.maximum_bounding_box_aspect_ratio = 1.5;
    const auto unfiltered = detect_bright_target(frame, configuration);
    ASSERT_TRUE(unfiltered.has_value());
    EXPECT_LT(unfiltered->center_x, 50.0);
    configuration.minimum_circularity = 0.55;
    const auto filtered = detect_bright_target(frame, configuration);
    ASSERT_TRUE(filtered.has_value());
    EXPECT_NEAR(filtered->center_x, 70.0, 1e-12);
    EXPECT_NEAR(filtered->center_y, 55.0, 1e-12);
}

TEST(TargetDetection, CircularityHandlesDegenerateContoursAndInvalidBounds)
{
    cv::Mat frame{8, 8, CV_8UC1, cv::Scalar{0}};
    frame.at<std::uint8_t>(2, 3) = 255U;
    TargetDetectorConfig configuration{};
    configuration.minimum_circularity = 0.1;
    EXPECT_FALSE(detect_bright_target(frame, configuration).has_value());
    for (const double invalid : {0.0, -1.0, 1.1, std::numeric_limits<double>::infinity(),
                                 std::numeric_limits<double>::quiet_NaN()})
    {
        configuration.minimum_circularity = invalid;
        EXPECT_THROW(static_cast<void>(detect_bright_target(frame, configuration)), std::invalid_argument);
    }
}

TEST(TargetDetection, OtsuThresholdAdaptsToDimScene)
{
    const auto frame = make_synthetic_target_frame({96, 72, 48, 36, 8, 20U, 90U});
    TargetDetectorConfig configuration{};
    configuration.threshold = 125U;
    configuration.minimum_pixel_area = 50;
    EXPECT_FALSE(detect_bright_target(frame, configuration).has_value());
    configuration.threshold_mode = ThresholdMode::Otsu;
    const auto detection = detect_bright_target(frame, configuration);
    ASSERT_TRUE(detection.has_value());
    EXPECT_NEAR(detection->center_x, 48.0, 1e-12);
    EXPECT_NEAR(detection->center_y, 36.0, 1e-12);
    EXPECT_GE(detection->pixel_area, 50.0);
}

TEST(TargetDetection, RejectsUnknownThresholdStrategy)
{
    const cv::Mat frame{8, 8, CV_8UC1, cv::Scalar{0}};
    TargetDetectorConfig configuration{};
    configuration.threshold_mode = static_cast<ThresholdMode>(99);
    EXPECT_THROW(static_cast<void>(detect_bright_target(frame, configuration)), std::invalid_argument);
}

TEST(TargetDetection, SearchRegionRejectsDistantLargerDistractor)
{
    auto frame = make_synthetic_target_frame({100, 80, 70, 50, 5, 10U, 240U});
    const auto clean = detect_bright_target(frame, 100U, 20);
    cv::rectangle(frame, cv::Point{5, 5}, cv::Point{24, 14}, cv::Scalar{240}, cv::FILLED);
    TargetDetectorConfig configuration{};
    configuration.threshold = 100U;
    configuration.minimum_pixel_area = 20;
    const auto unrestricted = detect_bright_target(frame, configuration);
    ASSERT_TRUE(unrestricted.has_value());
    EXPECT_NEAR(unrestricted->center_x, 14.5, 1e-12);
    EXPECT_NEAR(unrestricted->center_y, 9.5, 1e-12);
    configuration.search_region = cv::Rect{55, 35, 31, 31};
    const auto localized = detect_bright_target(frame, configuration);
    ASSERT_TRUE(clean.has_value());
    ASSERT_TRUE(localized.has_value());
    EXPECT_NEAR(localized->center_x, 70.0, 1e-12);
    EXPECT_NEAR(localized->center_y, 50.0, 1e-12);
    EXPECT_DOUBLE_EQ(localized->pixel_area, clean->pixel_area);
}

TEST(TargetDetection, RejectsEmptyAndOutOfBoundsSearchRegions)
{
    const cv::Mat frame{48, 64, CV_8UC1, cv::Scalar{0}};
    TargetDetectorConfig configuration{};
    for (const auto& region : {
             cv::Rect{0, 0, 0, 1}, cv::Rect{-1, 0, 1, 1}, cv::Rect{50, 30, 20, 20},
             cv::Rect{std::numeric_limits<int>::max(), 0, std::numeric_limits<int>::max(), 1},
             cv::Rect{0, std::numeric_limits<int>::max(), 1, std::numeric_limits<int>::max()}})
    {
        configuration.search_region = region;
        EXPECT_THROW(static_cast<void>(detect_bright_target(frame, configuration)), std::invalid_argument);
    }
}

TEST(TargetDetection, ReturnsEveryAdmissibleCandidateWithoutChangingLegacySelection)
{
    auto frame = make_synthetic_target_frame({100, 80, 70, 50, 5, 10U, 240U});
    cv::rectangle(frame, cv::Rect{5, 5, 20, 10}, cv::Scalar{240}, cv::FILLED);
    TargetDetectorConfig configuration{};
    configuration.threshold = 100U;
    configuration.minimum_pixel_area = 20;
    const auto candidates = detect_bright_target_candidates(frame, configuration);
    ASSERT_EQ(candidates.size(), 2U);
    const auto target = std::find_if(candidates.cbegin(), candidates.cend(),
        [](const TargetDetection& candidate)
        {
            return std::abs(candidate.center_x - 70.0) < 1e-12
                && std::abs(candidate.center_y - 50.0) < 1e-12;
        });
    EXPECT_NE(target, candidates.cend());
    const auto largest = detect_bright_target(frame, configuration);
    ASSERT_TRUE(largest.has_value());
    EXPECT_NEAR(largest->center_x, 14.5, 1e-12);
    EXPECT_NEAR(largest->center_y, 9.5, 1e-12);
    configuration.search_region = cv::Rect{55, 35, 31, 31};
    const auto localized = detect_bright_target_candidates(frame, configuration);
    ASSERT_EQ(localized.size(), 1U);
    EXPECT_NEAR(localized.front().center_x, 70.0, 1e-12);
    EXPECT_NEAR(localized.front().center_y, 50.0, 1e-12);
    configuration.threshold = 255U;
    EXPECT_TRUE(detect_bright_target_candidates(frame, configuration).empty());
    EXPECT_FALSE(detect_bright_target(frame, configuration).has_value());
}

TEST(TargetDetection, SmoothsSensorNoiseWithoutMutatingSource)
{
    auto frame = make_synthetic_target_frame({100, 80, 50, 40, 5, 20U, 220U});
    frame.at<std::uint8_t>(40, 50) = 20U;
    const cv::Mat original = frame.clone();
    TargetDetectorConfig configuration;
    configuration.smoothing_kernel_size = 3;
    configuration.minimum_pixel_area = 20;
    const auto detection = detect_bright_target(frame, configuration);
    ASSERT_TRUE(detection.has_value());
    EXPECT_NEAR(detection->center_x, 50.0, 0.1);
    EXPECT_EQ(cv::norm(frame, original, cv::NORM_INF), 0.0);
    configuration.smoothing_kernel_size = 2;
    EXPECT_THROW(static_cast<void>(detect_bright_target(frame, configuration)), std::invalid_argument);
}
