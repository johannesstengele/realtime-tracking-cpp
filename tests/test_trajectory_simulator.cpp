#include "realtime_tracking/constant_velocity_model.hpp"
#include "realtime_tracking/state2d.hpp"
#include "realtime_tracking/trajectory_simulator.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

int main()
{
    int failures{};

    for (const double time_step : {
             0.0,
             -1.0,
             std::numeric_limits<double>::infinity(),
             -std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::quiet_NaN()})
    {
        bool rejected{false};

        try
        {
            static_cast<void>(ConstantVelocityModel{time_step});
        }
        catch (const std::invalid_argument&)
        {
            rejected = true;
        }

        if (!rejected)
        {
            std::cerr << "Accepted invalid time step: " << time_step << '\n';
            ++failures;
        }
    }

    const State2D initial{10.0, 20.0, 3.0, -1.0};
    const ConstantVelocityModel model{0.25};

    for (const std::size_t steps : {0U, 1U, 20U})
    {
        const auto trajectory = simulate_trajectory(model, initial, steps);

        if (trajectory.size() != steps + 1)
        {
            std::cerr << "Incorrect trajectory length.\n";
            ++failures;
            continue;
        }

        for (std::size_t index{}; index < trajectory.size(); ++index)
        {
            const State2D& state = trajectory[index];
            const double time = 0.25 * static_cast<double>(index);

            if (std::abs(state.x - (initial.x + initial.vx * time)) > 1e-12
                || std::abs(state.y - (initial.y + initial.vy * time)) > 1e-12
                || state.vx != initial.vx
                || state.vy != initial.vy)
            {
                std::cerr << "Incorrect state at step " << index << '\n';
                ++failures;
            }
        }
    }

    for (const std::size_t steps : {
             std::vector<State2D>{}.max_size(),
             std::numeric_limits<std::size_t>::max()})
    {
        bool rejected{false};

        try
        {
            static_cast<void>(simulate_trajectory(model, initial, steps));
        }
        catch (const std::length_error&)
        {
            rejected = true;
        }

        if (!rejected)
        {
            std::cerr << "Accepted impossible trajectory length.\n";
            ++failures;
        }
    }

    const auto truth = simulate_trajectory(model, initial, 100);
    const auto exact = add_position_noise(truth, 0.0, 42);
    const auto noisy = add_position_noise(truth, 2.0, 42);
    const auto repeated = add_position_noise(truth, 2.0, 42);
    const auto different = add_position_noise(truth, 2.0, 43);
    bool changed_seed{false};
    bool independent_coordinates{false};

    for (std::size_t index{}; index < truth.size(); ++index)
    {
        if (exact[index].x != truth[index].x || exact[index].y != truth[index].y
            || noisy[index].vx != truth[index].vx || noisy[index].vy != truth[index].vy
            || noisy[index].x != repeated[index].x || noisy[index].y != repeated[index].y)
        {
            std::cerr << "Noise copy, velocity or reproducibility contract failed.\n";
            ++failures;
        }

        changed_seed = changed_seed || noisy[index].x != different[index].x;
        independent_coordinates = independent_coordinates
            || noisy[index].x - truth[index].x != noisy[index].y - truth[index].y;
    }

    if (!changed_seed || !independent_coordinates || !add_position_noise({}, 1.0, 42).empty())
    {
        std::cerr << "Noise seed, independence or empty-input contract failed.\n";
        ++failures;
    }

    for (const double sigma : {-1.0, std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::quiet_NaN()})
    {
        bool rejected{false};

        try
        {
            static_cast<void>(add_position_noise(truth, sigma, 42));
        }
        catch (const std::invalid_argument&)
        {
            rejected = true;
        }

        if (!rejected)
        {
            std::cerr << "Accepted invalid noise deviation.\n";
            ++failures;
        }
    }

    return (failures == 0) ? 0 : 1;
}
