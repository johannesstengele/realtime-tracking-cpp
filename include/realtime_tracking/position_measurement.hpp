#pragma once

#include "realtime_tracking/gaussian_state.hpp"

#include <Eigen/Core>

struct PositionInnovation
{
    Eigen::Vector2d predicted_measurement;
    Eigen::Vector2d innovation;
    Eigen::Matrix2d innovation_covariance;
};

[[nodiscard]] PositionInnovation compute_position_innovation(
    const GaussianState& predicted_state,
    const Eigen::Vector2d& measurement,
    const Eigen::Matrix2d& measurement_covariance);

[[nodiscard]] GaussianState correct_position_measurement(
    const GaussianState& predicted_state,
    const Eigen::Vector2d& measurement,
    const Eigen::Matrix2d& measurement_covariance);
