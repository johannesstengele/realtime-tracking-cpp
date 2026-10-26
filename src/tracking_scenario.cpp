#include "realtime_tracking/tracking_scenario.hpp"

#include "realtime_tracking/image_noise.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <stdexcept>

namespace
{
void draw_target(cv::Mat& image, double x, double y, int radius, double intensity)
{
    constexpr int scale{256};
    cv::circle(image, {static_cast<int>(std::lround(x * scale)),
        static_cast<int>(std::lround(y * scale))}, radius * scale,
        cv::Scalar{intensity}, cv::FILLED, cv::LINE_8, 8);
}
}

State2D project_fixed_depth(const State2D& world, const PlanarCamera& camera)
{
    if (!std::isfinite(world.x) || !std::isfinite(world.y)
        || !std::isfinite(world.vx) || !std::isfinite(world.vy)
        || !std::isfinite(camera.focal_x) || camera.focal_x <= 0.0
        || !std::isfinite(camera.focal_y) || camera.focal_y <= 0.0
        || !std::isfinite(camera.principal_x) || !std::isfinite(camera.principal_y)
        || !std::isfinite(camera.depth) || camera.depth <= 0.0)
    {
        throw std::invalid_argument{"Camera and world state must be finite with positive focal lengths and depth."};
    }
    const double scale_x = camera.focal_x / camera.depth;
    const double scale_y = camera.focal_y / camera.depth;
    const State2D projected{camera.principal_x + scale_x * world.x,
        camera.principal_y + scale_y * world.y, scale_x * world.vx, scale_y * world.vy};
    if (!std::isfinite(projected.x) || !std::isfinite(projected.y)
        || !std::isfinite(projected.vx) || !std::isfinite(projected.vy))
    {
        throw std::overflow_error{"Projected state exceeds the finite numeric range."};
    }
    return projected;
}

std::vector<TrackingScenario> tracking_scenarios()
{
    return {TrackingScenario::Nominal, TrackingScenario::Maneuver, TrackingScenario::Clutter,
        TrackingScenario::ShortOcclusion, TrackingScenario::LongDropout,
        TrackingScenario::Degraded, TrackingScenario::Boundary};
}

std::string_view scenario_name(TrackingScenario scenario)
{
    switch (scenario)
    {
    case TrackingScenario::Nominal: return "nominal";
    case TrackingScenario::Maneuver: return "maneuver";
    case TrackingScenario::Clutter: return "clutter";
    case TrackingScenario::ShortOcclusion: return "short_occlusion";
    case TrackingScenario::LongDropout: return "long_dropout";
    case TrackingScenario::Degraded: return "degraded";
    case TrackingScenario::Boundary: return "boundary";
    }
    throw std::invalid_argument{"Unknown tracking scenario."};
}

TrackingScenario parse_scenario(std::string_view name)
{
    for (const auto scenario : tracking_scenarios())
    {
        if (scenario_name(scenario) == name)
        {
            return scenario;
        }
    }
    throw std::invalid_argument{"Unknown tracking scenario name."};
}

ScenarioFrame render_tracking_frame(const ScenarioConfig& configuration, int index)
{
    static_cast<void>(scenario_name(configuration.scenario));
    if (configuration.frame_count <= 0 || configuration.frame_count > 10'000
        || index < 0 || index >= configuration.frame_count
        || !std::isfinite(configuration.delta_time) || configuration.delta_time <= 0.0
        || configuration.delta_time > 1.0)
    {
        throw std::invalid_argument{"Scenario requires 1..10000 frames, a valid index and a time step in (0, 1]."};
    }
    const double time = static_cast<double>(index) * configuration.delta_time;
    State2D world{-3.0 + time, -1.0 + 0.3 * time, 1.0, 0.3};
    if (configuration.scenario == TrackingScenario::Maneuver)
    {
        world.y = -0.5 + 0.375 * std::sin(1.5 * time);
        world.vy = 0.5625 * std::cos(1.5 * time);
    }
    if (configuration.scenario == TrackingScenario::Boundary)
    {
        world.x -= 0.925;
    }
    const auto truth = project_fixed_depth(world, {});
    bool visible = truth.x > -5.0 && truth.x < 325.0 && truth.y > -5.0 && truth.y < 245.0;
    if (configuration.scenario == TrackingScenario::ShortOcclusion && index >= 45 && index < 48)
    {
        visible = false;
    }
    if (configuration.scenario == TrackingScenario::LongDropout && index >= 45 && index < 57)
    {
        visible = false;
    }
    const bool degraded = configuration.scenario == TrackingScenario::Degraded;
    const double background = degraded ? 50.0 + 20.0 * std::sin(time) : 20.0;
    cv::Mat clean(240, 320, CV_8UC1, cv::Scalar{background});
    if (visible)
    {
        draw_target(clean, truth.x, truth.y, 5, degraded ? 165.0 : 220.0);
    }
    if (configuration.scenario == TrackingScenario::Clutter && index >= 10)
    {
        draw_target(clean, truth.x + 16.0, truth.y, 4, 230.0);
        draw_target(clean, 270.0, 180.0, 7, 230.0);
    }
    if (degraded)
    {
        cv::GaussianBlur(clean, clean, {3, 3}, 0.8);
    }
    const double sigma = degraded ? 35.0 : 30.0;
    constexpr std::uint64_t run_multiplier{0x9E3779B97F4A7C15ULL};
    constexpr std::uint64_t frame_multiplier{0xBF58476D1CE4E5B9ULL};
    const std::uint64_t frame_seed = configuration.seed * run_multiplier
        + static_cast<std::uint64_t>(index) * frame_multiplier;
    return {add_gaussian_sensor_noise(clean, sigma, frame_seed), truth, visible};
}
