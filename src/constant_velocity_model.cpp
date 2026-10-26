#include "realtime_tracking/constant_velocity_model.hpp"
#include "realtime_tracking/state_transition.hpp"

#include <cmath>
#include <stdexcept>

ConstantVelocityModel::ConstantVelocityModel(const double dt)
    : dt_{dt}
{
    const bool is_invalid = !std::isfinite(dt) || dt <= 0.0;

    if (is_invalid)
    {
        throw std::invalid_argument{"Time step must be finite and strictly positive."};
    }
}

ConstantVelocityProcessModel make_constant_velocity_model(
    double time_step,
    double acceleration_spectral_density)
{
    const auto transition = constant_velocity_transition(time_step);
    if (!std::isfinite(acceleration_spectral_density) || acceleration_spectral_density < 0.0)
    {
        throw std::invalid_argument{"Acceleration spectral density must be finite and nonnegative."};
    }

    Eigen::Matrix4d covariance = Eigen::Matrix4d::Zero();
    if (acceleration_spectral_density > 0.0)
    {
        const double velocity_variance = acceleration_spectral_density * time_step;
        const double cross_covariance = velocity_variance * (time_step / 2.0);
        const double position_variance = (velocity_variance * (time_step / 3.0)) * time_step;
        covariance(0, 0) = position_variance;
        covariance(1, 1) = position_variance;
        covariance(0, 2) = cross_covariance;
        covariance(2, 0) = cross_covariance;
        covariance(1, 3) = cross_covariance;
        covariance(3, 1) = cross_covariance;
        covariance(2, 2) = velocity_variance;
        covariance(3, 3) = velocity_variance;
    }

    if (!covariance.allFinite())
    {
        throw std::overflow_error{"Process covariance exceeds the finite numeric range."};
    }

    return ConstantVelocityProcessModel{transition, covariance};
}

State2D ConstantVelocityModel::propagate(const State2D& state) const{
    return State2D{
        state.x + state.vx * dt_,
        state.y + state.vy * dt_,
        state.vx,
        state.vy
    };
}

