#pragma once

#include "realtime_tracking/state2d.hpp"

#include <Eigen/Core>

class ConstantVelocityModel{
public:
    explicit ConstantVelocityModel(const double dt);

    State2D propagate(const State2D& state) const;
    
private:
    const double dt_;
};

struct ConstantVelocityProcessModel
{
    Eigen::Matrix4d transition;
    Eigen::Matrix4d process_covariance;
};

[[nodiscard]] ConstantVelocityProcessModel make_constant_velocity_model(
    double time_step,
    double acceleration_spectral_density);
