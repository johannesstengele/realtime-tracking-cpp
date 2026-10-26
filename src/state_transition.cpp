#include "realtime_tracking/state_transition.hpp"

#include <cmath>
#include <stdexcept>

Eigen::Matrix4d constant_velocity_transition(double delta_time)
{
    if (!std::isfinite(delta_time) || delta_time <= 0.0)
    {
        throw std::invalid_argument{"Delta time must be finite and positive."};
    }

    Eigen::Matrix4d transition = Eigen::Matrix4d::Identity();
    transition(0, 2) = delta_time;
    transition(1, 3) = delta_time;
    return transition;
}
