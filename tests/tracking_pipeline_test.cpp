#include "realtime_tracking/kalman_target_tracker.hpp"
#include "realtime_tracking/tracking_scenario.hpp"

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include <cmath>

TEST(TrackingPipeline, MeetsScenarioAccuracyAndLifecycleRequirements)
{
    cv::setNumThreads(1);
    for (const auto scenario : tracking_scenarios())
    {
        SCOPED_TRACE(std::string{scenario_name(scenario)});
        KalmanTrackerConfig configuration;
        configuration.detector.threshold = 110U;
        configuration.detector.minimum_pixel_area = 20;
        configuration.detector.opening_kernel_size = 3;
        configuration.detector.smoothing_kernel_size = 3;
        configuration.detector.minimum_circularity = 0.45;
        configuration.collect_timings = true;
        KalmanTargetTracker tracker{configuration};
        ScenarioConfig simulation{scenario, 42U, 120, 1.0 / 30.0};
        int visible{};
        int correct{};
        double squared_error{};
        std::uint64_t initial_identity{};
        for (int index{}; index < simulation.frame_count; ++index)
        {
            const auto frame = render_tracking_frame(simulation, index);
            const auto update = tracker.update(frame.image, simulation.delta_time);
            if (index == 0) initial_identity = update.track_id;
            if (frame.visible)
            {
                ++visible;
                if (update.detection.has_value())
                {
                    const double dx = update.detection->center_x - frame.truth.x;
                    const double dy = update.detection->center_y - frame.truth.y;
                    if (dx * dx + dy * dy <= 9.0)
                    {
                        ++correct;
                        ASSERT_TRUE(update.state.has_value());
                        squared_error += std::pow(update.state->mean(0) - frame.truth.x, 2)
                            + std::pow(update.state->mean(1) - frame.truth.y, 2);
                    }
                }
            }
            if (scenario == TrackingScenario::ShortOcclusion && index >= 45 && index < 48)
            {
                EXPECT_EQ(update.status, TrackStatus::Coasting);
                EXPECT_EQ(update.track_id, initial_identity);
            }
            if (scenario == TrackingScenario::ShortOcclusion && index == 48)
            {
                EXPECT_EQ(update.status, TrackStatus::Tracking);
                EXPECT_EQ(update.track_id, initial_identity);
            }
            if (scenario == TrackingScenario::LongDropout && index == 50)
            {
                EXPECT_EQ(update.status, TrackStatus::Lost);
            }
            if (scenario == TrackingScenario::LongDropout && index == 57)
            {
                EXPECT_EQ(update.status, TrackStatus::Tracking);
                EXPECT_GT(update.track_id, initial_identity);
            }
            EXPECT_GT(update.timings.total_ns, 0.0);
            EXPECT_GE(update.timings.total_ns, update.timings.detection_ns);
        }
        ASSERT_GT(correct, 0);
        EXPECT_GE(static_cast<double>(correct) / visible, scenario == TrackingScenario::Nominal ? 0.99 : 0.95);
        EXPECT_LT(std::sqrt(squared_error / correct), scenario == TrackingScenario::Nominal ? 1.0 : 3.0);
    }
}
