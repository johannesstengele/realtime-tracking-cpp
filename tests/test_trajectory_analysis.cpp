#include "realtime_tracking/state2d.hpp"
#include "realtime_tracking/trajectory_analysis.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>

int main()
{
    int failures{};
    const std::array<State2D, 2> reference{
        State2D{0.0, 0.0, 1.0, 2.0},
        State2D{2.0, 3.0, 1.0, 2.0}
    };
    const std::array<State2D, 2> estimate{
        State2D{3.0, 4.0, -1.0, -2.0},
        State2D{2.0, 3.0, -1.0, -2.0}
    };

    if (std::abs(position_rmse(reference, estimate) - std::sqrt(12.5)) > 1e-12
        || position_rmse(reference, reference) != 0.0
        || position_rmse(std::span{reference}.last(1), std::span{estimate}.last(1)) != 0.0)
    {
        std::cerr << "Incorrect position RMSE for array or span input.\n";
        ++failures;
    }

    for (const double magnitude : {1e200, 1e-200})
    {
        const std::array<State2D, 2> origin{};
        const std::array<State2D, 2> offset{
            State2D{magnitude, magnitude, 0.0, 0.0},
            State2D{0.0, 0.0, 0.0, 0.0}
        };
        const double rmse = position_rmse(origin, offset);

        if (!std::isfinite(rmse) || std::abs(rmse / magnitude - 1.0) > 1e-12)
        {
            std::cerr << "RMSE lost range at magnitude " << magnitude << '\n';
            ++failures;
        }
    }

    const auto expect_invalid = [&failures](
        std::span<const State2D> truth,
        std::span<const State2D> prediction)
    {
        bool rejected{false};

        try
        {
            static_cast<void>(position_rmse(truth, prediction));
        }
        catch (const std::invalid_argument&)
        {
            rejected = true;
        }

        if (!rejected)
        {
            std::cerr << "Accepted invalid RMSE input.\n";
            ++failures;
        }
    };

    expect_invalid({}, {});
    expect_invalid(reference, std::span{estimate}.first(1));

    for (const double invalid : {
             std::numeric_limits<double>::infinity(),
             -std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::quiet_NaN()})
    {
        auto invalid_x = reference;
        auto invalid_y = reference;
        invalid_x[0].x = invalid;
        invalid_y[1].y = invalid;
        expect_invalid(invalid_x, reference);
        expect_invalid(reference, invalid_x);
        expect_invalid(invalid_y, reference);
        expect_invalid(reference, invalid_y);
    }

    const double maximum = std::numeric_limits<double>::max();
    const std::array<State2D, 4> extreme_reference{
        State2D{maximum, 0.0, 0.0, 0.0},
        State2D{}, State2D{}, State2D{}
    };
    const std::array<State2D, 4> extreme_estimate{
        State2D{-maximum, 0.0, 0.0, 0.0},
        State2D{}, State2D{}, State2D{}
    };

    if (std::abs(position_rmse(extreme_reference, extreme_estimate) / maximum - 1.0) > 1e-12)
    {
        std::cerr << "Intermediate displacement overflow corrupted RMSE.\n";
        ++failures;
    }

    bool rejected_overflow{false};

    try
    {
        static_cast<void>(position_rmse(
            std::span{extreme_reference}.first(1),
            std::span{extreme_estimate}.first(1)));
    }
    catch (const std::overflow_error&)
    {
        rejected_overflow = true;
    }

    if (!rejected_overflow)
    {
        std::cerr << "Nonrepresentable RMSE was not rejected.\n";
        ++failures;
    }

    return (failures == 0) ? 0 : 1;
}
