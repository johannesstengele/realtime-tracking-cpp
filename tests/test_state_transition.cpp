#include "realtime_tracking/constant_velocity_model.hpp"
#include "realtime_tracking/state_transition.hpp"

#include <Eigen/Core>

#include <iostream>
#include <limits>
#include <stdexcept>

int main()
{
    int failures{};
    const Eigen::Vector4d state{10.0, 5.0, 3.0, -2.0};
    const Eigen::Vector4d prediction = constant_velocity_transition(0.5) * state;
    const auto propagated = ConstantVelocityModel{0.5}.propagate({10.0, 5.0, 3.0, -2.0});
    const Eigen::Vector4d expected{11.5, 4.0, 3.0, -2.0};
    const Eigen::Vector4d scalar{propagated.x, propagated.y, propagated.vx, propagated.vy};

    if (!prediction.isApprox(expected, 1e-12) || !prediction.isApprox(scalar, 1e-12)
        || !(constant_velocity_transition(0.3) * constant_velocity_transition(0.2))
                .isApprox(constant_velocity_transition(0.5), 1e-12))
    {
        std::cerr << "Transition violates expected motion or composition.\n";
        ++failures;
    }

    for (const double dt : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::quiet_NaN()})
    {
        bool rejected{false};
        try
        {
            static_cast<void>(constant_velocity_transition(dt));
        }
        catch (const std::invalid_argument&)
        {
            rejected = true;
        }
        if (!rejected)
        {
            std::cerr << "Accepted invalid transition time step.\n";
            ++failures;
        }
    }
    return (failures == 0) ? 0 : 1;
}
