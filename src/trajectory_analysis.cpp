#include "realtime_tracking/trajectory_analysis.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace
{
double squared_distance(
    const State2D& state,
    double target_x,
    double target_y)
{
    const double dx{(state.x - target_x)};
    const double dy{(state.y - target_y)};

    return pow(dx, 2) + pow(dy, 2);
}
}

std::optional<State2D> closest_state_to(
    const std::vector<State2D>& trajectory,
    double target_x,
    double target_y)
{
    if (trajectory.empty())
    {
        return std::nullopt;
    }

    if (!std::isfinite(target_x) || !std::isfinite(target_y))
    {
        throw std::invalid_argument{
            "Target coordinates must be finite."
        };
    }

    const auto closest = std::min_element(
        trajectory.cbegin(),
        trajectory.cend(),
        [target_x, target_y](
            const State2D& left,
            const State2D& right)
        {
            return squared_distance(left, target_x, target_y) 
                < squared_distance(right, target_x, target_y);
        });

    return *closest;
}

std::size_t count_states_within_radius(
    const std::vector<State2D>& trajectory,
    double center_x,
    double center_y,
    double radius)
{
    const bool has_invalid_geometry =
        !std::isfinite(center_x)
        || !std::isfinite(center_y)
        || !std::isfinite(radius)
        || radius < 0.0;

    if (has_invalid_geometry)
    {
        throw std::invalid_argument{
            "Detection region must be finite with nonnegative radius."
        };
    }

    const double radius_squared{radius * radius};

    const auto count = std::count_if(
        trajectory.cbegin(),
        trajectory.cend(),
        [center_x, center_y, radius_squared](
            const State2D& state)
        {
            return squared_distance(state, center_x, center_y) < radius_squared;
        });

    return static_cast<std::size_t>(count);
}

double position_rmse(
    std::span<const State2D> reference,
    std::span<const State2D> estimate
)
{
    if (reference.size() != estimate.size())
    {
        throw std::invalid_argument{
            "Reference and estimate are not equal size"
        };
    }

    if (reference.empty())
    {
        throw std::invalid_argument{
            "Trajectories are empty."
        };
    }

    double scale{};

    const auto include_difference = [&scale](double left, double right)
    {
        if (!std::isfinite(left) || !std::isfinite(right))
        {
            throw std::invalid_argument{
                "Trajectory positions must be finite."
            };
        }

        const double difference = left - right;
        const double magnitude = std::isfinite(difference)
            ? std::abs(difference)
            : std::max(std::abs(left), std::abs(right));

        scale = std::max(scale, magnitude);
    };

    for (std::size_t index{}; index < reference.size(); ++index)
    {
        include_difference(reference[index].x, estimate[index].x);
        include_difference(reference[index].y, estimate[index].y);
    }

    if (scale == 0.0)
    {
        return 0.0;
    }

    const auto normalized_difference = [scale](double left, double right)
    {
        const double difference = left - right;

        return std::isfinite(difference)
            ? difference / scale
            : left / scale - right / scale;
    };

    double sum_square_error{};

    for (std::size_t index{};
        index < reference.size();
        ++index
    )
    {
        const double dx = normalized_difference(reference[index].x, estimate[index].x);
        const double dy = normalized_difference(reference[index].y, estimate[index].y);

        sum_square_error += dx * dx + dy * dy;
    }

    const double rmse = scale * std::sqrt(
        sum_square_error / static_cast<double>(reference.size()));

    if (!std::isfinite(rmse))
    {
        throw std::overflow_error{
            "Position RMSE exceeds the finite numeric range."
        };
    }

    return rmse;
}
