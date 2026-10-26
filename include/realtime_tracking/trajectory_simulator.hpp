#pragma once

#include "realtime_tracking/constant_velocity_model.hpp"
#include "realtime_tracking/state2d.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

[[nodiscard]] std::vector<State2D> simulate_trajectory(
    const ConstantVelocityModel& model,
    const State2D& initial,
    std::size_t number_of_steps);

[[nodiscard]] std::vector<State2D> add_position_noise(
    std::span<const State2D> ground_truth,
    double sigma,
    std::uint32_t seed);
