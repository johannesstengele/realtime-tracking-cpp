#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <cmath>
#include <stdexcept>

namespace tracking_detail
{
template<int Dimension>
void validate_covariance(const Eigen::Matrix<double, Dimension, Dimension>& covariance)
{
    if (!covariance.allFinite() || covariance.diagonal().minCoeff() < 0.0)
    {
        throw std::invalid_argument{"Covariance must be finite with nonnegative variances."};
    }

    Eigen::Matrix<double, Dimension, 1> deviations;
    for (Eigen::Index index{}; index < Dimension; ++index)
    {
        deviations(index) = std::sqrt(covariance(index, index));
    }

    Eigen::Matrix<double, Dimension, Dimension> normalized =
        Eigen::Matrix<double, Dimension, Dimension>::Zero();
    for (Eigen::Index row{}; row < Dimension; ++row)
    {
        for (Eigen::Index column{}; column < Dimension; ++column)
        {
            if (deviations(row) == 0.0 || deviations(column) == 0.0)
            {
                if (covariance(row, column) != 0.0)
                {
                    throw std::invalid_argument{"Zero variance requires zero cross-covariance."};
                }
                continue;
            }
            const double correlation = covariance(row, column) / deviations(row) / deviations(column);
            if (!std::isfinite(correlation) || std::abs(correlation) > 1.0 + 1e-12)
            {
                throw std::invalid_argument{"Covariance implies an invalid correlation."};
            }
            normalized(row, column) = correlation;
        }
    }

    if (!normalized.isApprox(normalized.transpose(), 1e-12))
    {
        throw std::invalid_argument{"Covariance must be symmetric."};
    }
    const Eigen::LDLT<Eigen::Matrix<double, Dimension, Dimension>> factorization{normalized};

    if (factorization.info() != Eigen::Success
        || factorization.vectorD().minCoeff() < -1e-12)
    {
        throw std::invalid_argument{"Covariance must be positive semidefinite."};
    }
}
}
