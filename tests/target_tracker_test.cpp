#include "realtime_tracking/synthetic_target.hpp"
#include "realtime_tracking/target_tracker.hpp"

#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

#include <stdexcept>

namespace
{
TargetDetectorConfig detector_configuration()
{
    TargetDetectorConfig configuration{};
    configuration.threshold = 100U;
    configuration.minimum_pixel_area = 20;
    return configuration;
}
}

TEST(TargetTracker, UsesLocalSearchAndReacquiresAfterLargeMotion)
{
    TargetTracker tracker{detector_configuration(), {31, 31}};
    EXPECT_FALSE(tracker.has_track());
    const auto first = tracker.update(make_synthetic_target_frame({100, 80, 20, 20, 5, 10U, 240U}));
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(first->source, DetectionSource::InitialFullFrame);
    EXPECT_TRUE(tracker.has_track());
    const auto nearby = tracker.update(make_synthetic_target_frame({100, 80, 26, 22, 5, 10U, 240U}));
    ASSERT_TRUE(nearby.has_value());
    EXPECT_EQ(nearby->source, DetectionSource::LocalSearch);
    EXPECT_TRUE(tracker.has_track());
    const auto jumped = tracker.update(make_synthetic_target_frame({100, 80, 80, 50, 5, 10U, 240U}));
    ASSERT_TRUE(jumped.has_value());
    EXPECT_EQ(jumped->source, DetectionSource::Reacquisition);
    EXPECT_NEAR(jumped->detection.center_x, 80.0, 1e-12);
    EXPECT_NEAR(jumped->detection.center_y, 50.0, 1e-12);
    const cv::Mat empty_scene{80, 100, CV_8UC1, cv::Scalar{10}};
    EXPECT_FALSE(tracker.update(empty_scene).has_value());
    EXPECT_FALSE(tracker.has_track());
    const auto restored = tracker.update(make_synthetic_target_frame({100, 80, 20, 20, 5, 10U, 240U}));
    ASSERT_TRUE(restored.has_value());
    EXPECT_EQ(restored->source, DetectionSource::InitialFullFrame);
}

TEST(TargetTracker, LocalSuccessTakesPrecedenceOverDistantClutter)
{
    TargetTracker tracker{detector_configuration(), {31, 31}};
    ASSERT_TRUE(tracker.update(make_synthetic_target_frame({100, 80, 20, 20, 5, 10U, 240U})).has_value());
    auto next = make_synthetic_target_frame({100, 80, 26, 22, 5, 10U, 240U});
    cv::rectangle(next, cv::Rect{70, 50, 20, 10}, cv::Scalar{240}, cv::FILLED);
    const auto update = tracker.update(next);
    ASSERT_TRUE(update.has_value());
    EXPECT_EQ(update->source, DetectionSource::LocalSearch);
    EXPECT_NEAR(update->detection.center_x, 26.0, 1e-12);
    EXPECT_NEAR(update->detection.center_y, 22.0, 1e-12);
}

TEST(TargetTracker, InvalidUpdatesPreserveTrackAndResetClearsIt)
{
    TargetTracker tracker{detector_configuration(), {31, 31}};
    const auto frame = make_synthetic_target_frame({100, 80, 20, 20, 5, 10U, 240U});
    ASSERT_TRUE(tracker.update(frame).has_value());
    EXPECT_THROW(static_cast<void>(tracker.update({})), std::invalid_argument);
    EXPECT_TRUE(tracker.has_track());
    const cv::Mat too_small{20, 20, CV_8UC1, cv::Scalar{10}};
    EXPECT_THROW(static_cast<void>(tracker.update(too_small)), std::invalid_argument);
    EXPECT_TRUE(tracker.has_track());
    const auto continued = tracker.update(frame);
    ASSERT_TRUE(continued.has_value());
    EXPECT_EQ(continued->source, DetectionSource::LocalSearch);
    tracker.reset();
    EXPECT_FALSE(tracker.has_track());
    const auto restarted = tracker.update(frame);
    ASSERT_TRUE(restarted.has_value());
    EXPECT_EQ(restarted->source, DetectionSource::InitialFullFrame);
}

TEST(TargetTracker, RejectsInvalidConfiguration)
{
    auto configuration = detector_configuration();
    EXPECT_THROW(static_cast<void>(TargetTracker(configuration, cv::Size{30, 31})), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(TargetTracker(configuration, cv::Size{31, 0})), std::invalid_argument);
    configuration.search_region = cv::Rect{0, 0, 31, 31};
    EXPECT_THROW(static_cast<void>(TargetTracker(configuration, cv::Size{31, 31})), std::invalid_argument);
    configuration.search_region.reset();
    configuration.minimum_pixel_area = 0;
    EXPECT_THROW(static_cast<void>(TargetTracker(configuration, cv::Size{31, 31})), std::invalid_argument);
}

TEST(TargetTracker, ReacquiresAfterFrameSizeChanges)
{
    TargetTracker tracker{detector_configuration(), {31, 31}};
    ASSERT_TRUE(tracker.update(make_synthetic_target_frame({100, 80, 80, 50, 5, 10U, 240U})).has_value());
    const auto resized = tracker.update(make_synthetic_target_frame({50, 50, 20, 20, 5, 10U, 240U}));
    ASSERT_TRUE(resized.has_value());
    EXPECT_EQ(resized->source, DetectionSource::Reacquisition);
    EXPECT_NEAR(resized->detection.center_x, 20.0, 1e-12);
    EXPECT_NEAR(resized->detection.center_y, 20.0, 1e-12);
}
