#pragma once

#include <Eigen/Core>

#include <vector>

struct MeasurementStatistics
{
    Eigen::Vector2d mean_error;
    Eigen::Matrix2d covariance;
};

[[nodiscard]] MeasurementStatistics estimate_measurement_statistics(
    const std::vector<Eigen::Vector2d>& errors);
