#include "realtime_tracking/tracking_scenario.hpp"

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include <limits>
#include <stdexcept>

TEST(TrackingScenario, CameraProjectsPositionAndVelocity)
{
    const auto state = project_fixed_depth({-3.0, -1.0, 1.0, 0.3}, {});
    EXPECT_DOUBLE_EQ(state.x, 40.0);
    EXPECT_DOUBLE_EQ(state.y, 80.0);
    EXPECT_DOUBLE_EQ(state.vx, 40.0);
    EXPECT_DOUBLE_EQ(state.vy, 12.0);
    PlanarCamera camera;
    camera.depth = 0.0;
    EXPECT_THROW(static_cast<void>(project_fixed_depth({}, camera)), std::invalid_argument);
    camera.depth = 1.0;
    camera.focal_x = std::numeric_limits<double>::max();
    EXPECT_THROW(static_cast<void>(project_fixed_depth({10.0, 0.0, 0.0, 0.0}, camera)), std::overflow_error);
}

TEST(TrackingScenario, RenderingIsIndependentOfCallOrder)
{
    ScenarioConfig configuration;
    const auto first = render_tracking_frame(configuration, 15);
    static_cast<void>(render_tracking_frame(configuration, 2));
    const auto repeated = render_tracking_frame(configuration, 15);
    EXPECT_EQ(cv::norm(first.image, repeated.image, cv::NORM_INF), 0.0);
    ++configuration.seed;
    EXPECT_GT(cv::norm(first.image, render_tracking_frame(configuration, 15).image, cv::NORM_INF), 0.0);
}

TEST(TrackingScenario, EncodesVisibilityAndAnalyticManeuverVelocity)
{
    ScenarioConfig configuration;
    configuration.scenario = TrackingScenario::ShortOcclusion;
    EXPECT_TRUE(render_tracking_frame(configuration, 44).visible);
    EXPECT_FALSE(render_tracking_frame(configuration, 45).visible);
    EXPECT_TRUE(render_tracking_frame(configuration, 48).visible);
    configuration.scenario = TrackingScenario::LongDropout;
    EXPECT_FALSE(render_tracking_frame(configuration, 56).visible);
    EXPECT_TRUE(render_tracking_frame(configuration, 57).visible);
    configuration.scenario = TrackingScenario::Maneuver;
    const auto before = render_tracking_frame(configuration, 29).truth;
    const auto current = render_tracking_frame(configuration, 30).truth;
    const auto after = render_tracking_frame(configuration, 31).truth;
    EXPECT_NEAR(current.vy, (after.y - before.y) / (2.0 * configuration.delta_time), 0.02);
}

TEST(TrackingScenario, ValidatesScenarioAndFrameBounds)
{
    for (const auto scenario : tracking_scenarios())
    {
        EXPECT_EQ(parse_scenario(scenario_name(scenario)), scenario);
    }
    EXPECT_THROW(static_cast<void>(parse_scenario("missing")), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(render_tracking_frame({}, -1)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(render_tracking_frame({}, 120)), std::invalid_argument);
}
