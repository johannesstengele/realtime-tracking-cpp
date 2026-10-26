#include "realtime_tracking/constant_velocity_model.hpp"
#include "realtime_tracking/state2d.hpp"
#include "realtime_tracking/demonstrate_rejection.hpp"
#include "realtime_tracking/trajectory_simulator.hpp"
#include "realtime_tracking/trajectory_analysis.hpp"
#include "realtime_tracking/trajectory_io.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <cmath>
#include <iostream>
#include <stdexcept>

bool approximately_equal(
    double actual,
    double expected,
    double tolerance)
{
    return (std::abs(actual - expected) < tolerance);
}

int main()
{
    int failures{};

    const State2D initial{
        10.0,
        20.0,
        3.0,
        -1.0
    };

    const ConstantVelocityModel model{2.0};
    const State2D next = model.propagate(initial);

    const double tolerance{1e-12};

    if (!approximately_equal(next.x, 16.0, tolerance))
    {
        std::cerr << "Expected x = 16, received "
                  << next.x << '\n';
        ++failures;
    }

    if (!approximately_equal(next.y, 18.0, tolerance))
    {
        std::cerr << "Expected y = 18, received "
                  << next.y << '\n';
        ++failures;
    }

    bool rejected_zero{false};

    try
    {
        static_cast<void>(ConstantVelocityModel{0.0});
    }
    catch (const std::invalid_argument&)
    {
        rejected_zero = true;
    }

    if (!rejected_zero)
    {
        std::cerr << "Model accepted dt = 0.\n";
        ++failures;
    }

    const std::vector<State2D> trajectory = simulate_trajectory(model, initial, 3);

    if (trajectory.size() != 4)
    {
        std::cerr << "Unexpected trajectory size: "
                << trajectory.size() << '\n';
        ++failures;
    }

    const State2D& final_state = trajectory.back();

    if (!approximately_equal(
            final_state.x,
            28.0,
            tolerance))
    {
        std::cerr << "Unexpected final x: "
                << final_state.x << '\n';
        ++failures;
    }

    if (!approximately_equal(
            final_state.y,
            14.0,
            tolerance))
    {
        std::cerr << "Unexpected final y: "
                << final_state.y << '\n';
        ++failures;
    }

    const std::optional<State2D> closest =
    closest_state_to(trajectory, 21.0, 15.0);

    if (!closest.has_value())
    {
        std::cerr << "No closest state was returned.\n";
        ++failures;
    }
    else if (!approximately_equal(
                closest->x, 22.0, tolerance)
            || !approximately_equal(
                closest->y, 16.0, tolerance))
    {
        std::cerr << "Incorrect closest state: ("
                << closest->x << ", "
                << closest->y << ")\n";
        ++failures;
    }

    const std::size_t states_in_region =
        count_states_within_radius(
            trajectory,
            21.0,
            15.0,
            6.0);

    if (states_in_region != 2)
    {
        std::cerr << "Incorrect number of states in region: "
                << states_in_region << '\n';
        ++failures;
    }

    const std::vector<State2D> empty;

    const std::optional<State2D> empty_result =
        closest_state_to(empty, 0.0, 0.0);

    if (empty_result.has_value())
    {
        std::cerr
            << "Empty trajectory unexpectedly produced a state.\n";
        ++failures;
    }

    const std::filesystem::path csv_path{
        "trajectory_test.csv"
    };

    write_trajectory_csv(trajectory, csv_path);

    std::ifstream input{csv_path};

    if (!input)
    { 
        std::cerr << "Could not read generated CSV.\n";
        ++failures;
    }
    else
    {
        std::string line;
        std::size_t line_count{};

        while (std::getline(input, line))
        {
            ++line_count;
        }

        const std::size_t expected_lines = 4 + 1;

        if (line_count != expected_lines)
        {
            std::cerr
                << "Unexpected CSV line count: "
                << line_count << '\n';
            ++failures;
        }
    }

    input.close();

    if (!std::filesystem::remove(csv_path))
    {
        std::cerr << "Could not remove temporary CSV.\n";
        ++failures;
    }

    const std::vector<State2D> reference{
        State2D{0.0, 0.0, 0.0, 0.0},
        State2D{2.0, 3.0, 0.0, 0.0}
    };

    const std::vector<State2D> estimate{
        State2D{3.0, 4.0, 0.0, 0.0},
        State2D{2.0, 3.0, 0.0, 0.0}
    };

    const std::vector<State2D> longer{
        State2D(),
        State2D(),
        State2D()
    };

    const double rmse = position_rmse(reference, estimate);
    const double expected_rmse = std::sqrt(12.5);
    
    if (not approximately_equal(rmse, expected_rmse, 0.00001))
    {
        std::cout << "RMSE computation is wrong";
        ++failures;
    }

    bool accepted_different_lengths = true;
    try
    {
        static_cast<void>(position_rmse(reference, longer));
    }
    catch (const std::invalid_argument&)
    {
        std::cout << "Different length correctly catched.";
        accepted_different_lengths = false;
    }

    if (accepted_different_lengths)
    {
        ++failures;
    }

    return (failures == 0) ? 0 : 1;
}
