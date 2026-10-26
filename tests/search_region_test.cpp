#include "realtime_tracking/search_region.hpp"
#include "realtime_tracking/synthetic_target.hpp"
#include "realtime_tracking/target_detection.hpp"

#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

#include <limits>
#include <stdexcept>

TEST(SearchRegion, PreservesWindowSizeNearFrameBoundary)
{
    EXPECT_EQ(make_centered_search_region({2.0, 3.0}, {31, 31}, {100, 80}), cv::Rect(0, 0, 31, 31));
    EXPECT_EQ(make_centered_search_region({98.0, 78.0}, {31, 31}, {100, 80}), cv::Rect(69, 49, 31, 31));
    EXPECT_EQ(make_centered_search_region({70.0, 50.0}, {31, 31}, {100, 80}), cv::Rect(55, 35, 31, 31));
    EXPECT_EQ(make_centered_search_region({70.5, 50.5}, {31, 31}, {100, 80}), cv::Rect(56, 36, 31, 31));
}

TEST(SearchRegion, RejectsInvalidGeometry)
{
    EXPECT_THROW(static_cast<void>(make_centered_search_region({-0.1, 0.0}, {3, 3}, {10, 10})),
        std::invalid_argument);
    EXPECT_THROW(static_cast<void>(make_centered_search_region({10.0, 0.0}, {3, 3}, {10, 10})),
        std::invalid_argument);
    EXPECT_THROW(static_cast<void>(make_centered_search_region(
        {std::numeric_limits<double>::quiet_NaN(), 0.0}, {3, 3}, {10, 10})), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(make_centered_search_region({0.0, 0.0}, {2, 3}, {10, 10})),
        std::invalid_argument);
    EXPECT_THROW(static_cast<void>(make_centered_search_region({0.0, 0.0}, {0, 3}, {10, 10})),
        std::invalid_argument);
    EXPECT_THROW(static_cast<void>(make_centered_search_region({0.0, 0.0}, {11, 3}, {10, 10})),
        std::invalid_argument);
    EXPECT_THROW(static_cast<void>(make_centered_search_region({0.0, 0.0}, {3, 3}, {0, 10})),
        std::invalid_argument);
}

TEST(SearchRegion, PreviousDetectionRejectsDistantClutter)
{
    TargetDetectorConfig configuration{};
    configuration.threshold = 100U;
    configuration.minimum_pixel_area = 20;
    const auto first = detect_bright_target(
        make_synthetic_target_frame({100, 80, 70, 50, 5, 10U, 240U}), configuration);
    ASSERT_TRUE(first.has_value());
    auto frame = make_synthetic_target_frame({100, 80, 75, 52, 5, 10U, 240U});
    cv::rectangle(frame, cv::Point{2, 2}, cv::Point{21, 11}, cv::Scalar{240}, cv::FILLED);
    const auto unrestricted = detect_bright_target(frame, configuration);
    ASSERT_TRUE(unrestricted.has_value());
    EXPECT_NEAR(unrestricted->center_x, 11.5, 1e-12);
    EXPECT_NEAR(unrestricted->center_y, 6.5, 1e-12);
    configuration.search_region = make_centered_search_region(
        {first->center_x, first->center_y}, {31, 31}, frame.size());
    const auto tracked = detect_bright_target(frame, configuration);
    ASSERT_TRUE(tracked.has_value());
    EXPECT_NEAR(tracked->center_x, 75.0, 1e-12);
    EXPECT_NEAR(tracked->center_y, 52.0, 1e-12);
}
