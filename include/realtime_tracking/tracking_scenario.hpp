#pragma once

#include "realtime_tracking/state2d.hpp"

#include <opencv2/core/mat.hpp>

#include <cstdint>
#include <string_view>
#include <vector>

enum class TrackingScenario
{
    Nominal,
    Maneuver,
    Clutter,
    ShortOcclusion,
    LongDropout,
    Degraded,
    Boundary
};

struct PlanarCamera
{
    double focal_x{200.0};
    double focal_y{200.0};
    double principal_x{160.0};
    double principal_y{120.0};
    double depth{5.0};
};

struct ScenarioConfig
{
    TrackingScenario scenario{TrackingScenario::Nominal};
    std::uint64_t seed{1U};
    int frame_count{120};
    double delta_time{1.0 / 30.0};
};

struct ScenarioFrame
{
    cv::Mat image;
    State2D truth;
    bool visible;
};

[[nodiscard]] State2D project_fixed_depth(const State2D& world, const PlanarCamera& camera);
[[nodiscard]] std::string_view scenario_name(TrackingScenario scenario);
[[nodiscard]] TrackingScenario parse_scenario(std::string_view name);
[[nodiscard]] std::vector<TrackingScenario> tracking_scenarios();
[[nodiscard]] ScenarioFrame render_tracking_frame(const ScenarioConfig& configuration, int index);
