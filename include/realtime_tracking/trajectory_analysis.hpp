#pragma once

#include "realtime_tracking/state2d.hpp"

#include <cstddef>
#include <optional>
#include <vector>
#include <span>

[[nodiscard]] std::optional<State2D> closest_state_to(
    const std::vector<State2D>& trajectory,
    double target_x,
    double target_y);

[[nodiscard]] std::size_t count_states_within_radius(
    const std::vector<State2D>& trajectory,
    double center_x,
    double center_y,
    double radius);

[[nodiscard]] double position_rmse(
    std::span<const State2D> reference, 
    std::span<const State2D> estimate);