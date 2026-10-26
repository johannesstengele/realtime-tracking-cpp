#include "realtime_tracking/measurement_statistics.hpp"

#include <cmath>
#include <stdexcept>

MeasurementStatistics estimate_measurement_statistics(const std::vector<Eigen::Vector2d>& errors)
{
    if (errors.size() < 2U)
    {
        throw std::invalid_argument{"At least two errors are required."};
    }

    const double count = static_cast<double>(errors.size());
    Eigen::Vector2d mean = Eigen::Vector2d::Zero();
    for (const auto& error : errors)
    {
        if (!error.allFinite())
        {
            throw std::invalid_argument{"Measurement errors must be finite."};
        }
        mean += error / count;
    }

    const double normalization = std::sqrt(static_cast<double>(errors.size() - 1));
    Eigen::Matrix2d covariance = Eigen::Matrix2d::Zero();
    for (const auto& error : errors)
    {
        const Eigen::Vector2d centered = (error - mean) / normalization;
        covariance += centered * centered.transpose();
    }

    if (!mean.allFinite() || !covariance.allFinite())
    {
        throw std::overflow_error{"Measurement statistics exceed the finite numeric range."};
    }

    return MeasurementStatistics{mean, covariance};
}
