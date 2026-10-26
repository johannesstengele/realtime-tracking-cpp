#include "realtime_tracking/trajectory_simulator.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

std::vector<State2D> simulate_trajectory(
    const ConstantVelocityModel& model,
    const State2D& initial,
    std::size_t number_of_steps)
{
    std::vector<State2D> trajectory;

    if (number_of_steps >= trajectory.max_size())
    {
        throw std::length_error{
            "Trajectory exceeds the maximum number of states."
        };
    }

    trajectory.reserve(number_of_steps + 1);
    trajectory.push_back(initial);

    State2D current{initial};
    State2D next;

    for (std::size_t step{0};
         step < number_of_steps;
         ++step)
    {
        next = model.propagate(current);
        trajectory.push_back(next);
        current = next;
    }

    return trajectory;
}

std::vector<State2D> add_position_noise(
    std::span<const State2D> ground_truth,
    double sigma,
    std::uint32_t seed)
{
    if (!std::isfinite(sigma) || sigma < 0.0)
    {
        throw std::invalid_argument{
            "Noise standard deviation must be finite and nonnegative."
        };
    }

    std::vector<State2D> noisy{ground_truth.begin(), ground_truth.end()};

    if (sigma == 0.0)
    {
        return noisy;
    }

    std::mt19937 engine{seed};
    std::normal_distribution<double> noise{0.0, sigma};

    for (State2D& state : noisy)
    {
        if (!std::isfinite(state.x) || !std::isfinite(state.y))
        {
            throw std::invalid_argument{"Ground-truth positions must be finite."};
        }

        state.x += noise(engine);
        state.y += noise(engine);

        if (!std::isfinite(state.x) || !std::isfinite(state.y))
        {
            throw std::overflow_error{"Noisy position exceeds the finite numeric range."};
        }
    }

    return noisy;
}
