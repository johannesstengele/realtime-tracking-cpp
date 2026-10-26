#pragma once

#include "realtime_tracking/constant_velocity_model.hpp"

#include <Eigen/Core>

struct GaussianState
{
    Eigen::Vector4d mean;
    Eigen::Matrix4d covariance;
};

class GaussianPredictor
{
public:
    GaussianPredictor(double delta_time, double acceleration_sigma);

    [[nodiscard]] GaussianState predict(const GaussianState& current) const;

private:
    Eigen::Matrix4d transition_;
    Eigen::Matrix4d process_noise_;
};

[[nodiscard]] GaussianState predict_gaussian_state(
    const GaussianState& current,
    double delta_time,
    double acceleration_sigma);

[[nodiscard]] GaussianState predict_gaussian_state(
    const GaussianState& posterior,
    const ConstantVelocityProcessModel& model);
