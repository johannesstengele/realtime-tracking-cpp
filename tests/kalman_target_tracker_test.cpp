#include "realtime_tracking/kalman_target_tracker.hpp"

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include <limits>
#include <stdexcept>

namespace
{
cv::Mat frame_at(int x)
{
    cv::Mat frame(160, 240, CV_8UC1, cv::Scalar{20});
    if (x >= 0)
    {
        cv::circle(frame, {x, 60}, 4, cv::Scalar{220}, cv::FILLED);
    }
    return frame;
}
}

TEST(KalmanTargetTracker, EstimatesVelocityAndUsesPredictionDuringDropouts)
{
    KalmanTargetTracker tracker;
    KalmanTrackingUpdate update;
    for (int index{}; index < 15; ++index)
    {
        update = tracker.update(frame_at(20 + 3 * index), 0.1);
        ASSERT_EQ(update.status, TrackStatus::Tracking);
    }
    ASSERT_TRUE(update.state.has_value());
    EXPECT_NEAR(update.state->mean(2), 30.0, 0.5);
    const auto identity = update.track_id;
    for (int index{}; index < 3; ++index)
    {
        update = tracker.update(frame_at(-1), 0.1);
        EXPECT_EQ(update.status, TrackStatus::Coasting);
        ASSERT_TRUE(update.state.has_value());
        EXPECT_NEAR(update.state->mean(0), 65.0 + 3.0 * index, 0.5);
    }
    update = tracker.update(frame_at(74), 0.1);
    EXPECT_EQ(update.status, TrackStatus::Tracking);
    EXPECT_EQ(update.track_id, identity);
    EXPECT_TRUE(update.normalized_innovation_squared.has_value());
}

TEST(KalmanTargetTracker, SelectsPlausibleCandidateInsteadOfLargestBlob)
{
    KalmanTargetTracker tracker;
    for (int index{}; index < 10; ++index)
    {
        static_cast<void>(tracker.update(frame_at(20 + index), 0.1));
    }
    auto cluttered = frame_at(30);
    cv::circle(cluttered, {48, 60}, 7, cv::Scalar{230}, cv::FILLED);
    const auto result = tracker.update(cluttered, 0.1);
    ASSERT_TRUE(result.detection.has_value());
    EXPECT_NEAR(result.detection->center_x, 30.0, 0.01);
}

TEST(KalmanTargetTracker, RejectsOutlierAndStartsNewIdentityAfterCoastLimit)
{
    KalmanTrackerConfig configuration;
    configuration.maximum_coast_frames = 2U;
    KalmanTargetTracker tracker{configuration};
    const auto first = tracker.update(frame_at(20), 0.1);
    EXPECT_EQ(tracker.update(frame_at(200), 0.1).status, TrackStatus::Coasting);
    EXPECT_EQ(tracker.update(frame_at(-1), 0.1).status, TrackStatus::Coasting);
    EXPECT_EQ(tracker.update(frame_at(-1), 0.1).status, TrackStatus::Lost);
    EXPECT_FALSE(tracker.has_track());
    const auto next = tracker.update(frame_at(200), 0.1);
    EXPECT_EQ(next.status, TrackStatus::Tracking);
    EXPECT_GT(next.track_id, first.track_id);
}

TEST(KalmanTargetTracker, InvalidInputPreservesTrackAndResetPreservesIdentityUniqueness)
{
    KalmanTargetTracker tracker;
    const auto first = tracker.update(frame_at(20), 0.1);
    EXPECT_THROW(static_cast<void>(tracker.update(cv::Mat{}, 0.1)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(tracker.update(frame_at(20), 0.0)), std::invalid_argument);
    EXPECT_TRUE(tracker.has_track());
    tracker.reset();
    EXPECT_FALSE(tracker.has_track());
    EXPECT_GT(tracker.update(frame_at(20), 0.1).track_id, first.track_id);
}

TEST(KalmanTargetTracker, RejectsInvalidConfigurations)
{
    KalmanTrackerConfig configuration;
    configuration.measurement_covariance.setZero();
    EXPECT_THROW(KalmanTargetTracker{configuration}, std::invalid_argument);
    configuration = {};
    configuration.squared_gate_threshold = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(KalmanTargetTracker{configuration}, std::invalid_argument);
    configuration = {};
    configuration.search_window = {64, 63};
    EXPECT_THROW(KalmanTargetTracker{configuration}, std::invalid_argument);
}
