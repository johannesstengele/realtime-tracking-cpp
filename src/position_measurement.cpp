#include "realtime_tracking/position_measurement.hpp"
#include "covariance_validation.hpp"

#include <Eigen/Cholesky>

#include <stdexcept>

PositionInnovation compute_position_innovation(
    const GaussianState& predicted_state,
    const Eigen::Vector2d& measurement,
    const Eigen::Matrix2d& measurement_covariance)
{
    if (!predicted_state.mean.allFinite() || !measurement.allFinite())
    {
        throw std::invalid_argument{"State mean and measurement must be finite."};
    }
    tracking_detail::validate_covariance(predicted_state.covariance);
    tracking_detail::validate_covariance(measurement_covariance);

    const Eigen::Vector2d predicted_measurement = predicted_state.mean.head<2>();
    const Eigen::Vector2d innovation = measurement - predicted_measurement;
    Eigen::Matrix2d covariance = predicted_state.covariance.topLeftCorner<2, 2>() + measurement_covariance;
    covariance = (0.5 * covariance + 0.5 * covariance.transpose()).eval();
    if (!innovation.allFinite() || !covariance.allFinite())
    {
        throw std::overflow_error{"Position innovation exceeds the finite numeric range."};
    }

    return PositionInnovation{predicted_measurement, innovation, covariance};
}

GaussianState correct_position_measurement(
    const GaussianState& predicted_state,
    const Eigen::Vector2d& measurement,
    const Eigen::Matrix2d& measurement_covariance)
{
    const auto position_innovation = compute_position_innovation(
        predicted_state, measurement, measurement_covariance);
    const Eigen::LDLT<Eigen::Matrix2d> factorization{position_innovation.innovation_covariance};
    if (factorization.info() != Eigen::Success || !factorization.vectorD().allFinite()
        || factorization.vectorD().minCoeff() <= 0.0)
    {
        throw std::runtime_error{"Innovation covariance must be positive definite and numerically solvable."};
    }

    const Eigen::Matrix<double, 4, 2> cross_covariance = predicted_state.covariance.leftCols<2>();
    const Eigen::Matrix<double, 4, 2> gain =
        factorization.solve(cross_covariance.transpose()).transpose();
    if (!gain.allFinite())
    {
        throw std::overflow_error{"Kalman gain exceeds the finite numeric range."};
    }

    Eigen::Matrix<double, 2, 4> observation = Eigen::Matrix<double, 2, 4>::Zero();
    observation(0, 0) = 1.0;
    observation(1, 1) = 1.0;
    const Eigen::Matrix4d residual_transform = Eigen::Matrix4d::Identity() - gain * observation;
    GaussianState corrected{
        predicted_state.mean + gain * position_innovation.innovation,
        residual_transform * predicted_state.covariance * residual_transform.transpose()
            + gain * measurement_covariance * gain.transpose()
    };
    corrected.covariance = (0.5 * corrected.covariance + 0.5 * corrected.covariance.transpose()).eval();
    if (!corrected.mean.allFinite() || !corrected.covariance.allFinite())
    {
        throw std::overflow_error{"Corrected state exceeds the finite numeric range."};
    }

    return corrected;
}
