#include "realtime_tracking/process_noise.hpp"

#include <cmath>
#include <stdexcept>

Eigen::Matrix4d constant_acceleration_process_noise(
    double delta_time,
    double acceleration_sigma)
{
    if (!std::isfinite(delta_time) || delta_time <= 0.0
        || !std::isfinite(acceleration_sigma) || acceleration_sigma < 0.0)
    {
        throw std::invalid_argument{
            "Time step must be finite and positive; acceleration deviation must be finite and nonnegative."
        };
    }

    if (acceleration_sigma == 0.0)
    {
        return Eigen::Matrix4d::Zero();
    }

    const double velocity_scale = acceleration_sigma * delta_time;
    const double position_scale = velocity_scale * (delta_time / 2.0);
    Eigen::Matrix<double, 4, 2> influence = Eigen::Matrix<double, 4, 2>::Zero();
    influence(0, 0) = position_scale;
    influence(1, 1) = position_scale;
    influence(2, 0) = velocity_scale;
    influence(3, 1) = velocity_scale;
    const Eigen::Matrix4d covariance = influence * influence.transpose();

    if (!covariance.allFinite())
    {
        throw std::overflow_error{"Process covariance exceeds the finite numeric range."};
    }

    return covariance;
}
