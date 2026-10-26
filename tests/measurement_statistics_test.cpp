#include "realtime_tracking/measurement_statistics.hpp"

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <limits>
#include <stdexcept>
#include <vector>

TEST(MeasurementStatistics, ComputesUnbiasedTwoDimensionalCovariance)
{
    const auto result = estimate_measurement_statistics({{1.0, 0.0}, {-1.0, 0.0}, {0.0, 1.0}, {0.0, -1.0}});
    EXPECT_TRUE(result.mean_error.isZero(1e-12));
    const Eigen::Matrix2d expected = Eigen::Matrix2d::Identity() * (2.0 / 3.0);
    EXPECT_TRUE(result.covariance.isApprox(expected, 1e-12));
}

TEST(MeasurementStatistics, SeparatesBiasFromCorrelatedVariance)
{
    const auto result = estimate_measurement_statistics({{1.0, 2.0}, {2.0, 4.0}, {3.0, 6.0}, {4.0, 8.0}});
    EXPECT_TRUE(result.mean_error.isApprox(Eigen::Vector2d{2.5, 5.0}, 1e-12));
    Eigen::Matrix2d expected;
    expected << 5.0 / 3.0, 10.0 / 3.0, 10.0 / 3.0, 20.0 / 3.0;
    EXPECT_TRUE(result.covariance.isApprox(expected, 1e-12));
    const auto constant = estimate_measurement_statistics({{3.0, -2.0}, {3.0, -2.0}});
    EXPECT_TRUE(constant.mean_error.isApprox(Eigen::Vector2d{3.0, -2.0}, 1e-12));
    EXPECT_TRUE(constant.covariance.isZero());
}

TEST(MeasurementStatistics, RejectsInsufficientAndNonfiniteErrors)
{
    EXPECT_THROW(static_cast<void>(estimate_measurement_statistics({})), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(estimate_measurement_statistics({{0.0, 0.0}})), std::invalid_argument);
    for (const double invalid : {std::numeric_limits<double>::quiet_NaN(),
                                 std::numeric_limits<double>::infinity()})
    {
        EXPECT_THROW(static_cast<void>(estimate_measurement_statistics({{invalid, 0.0}, {0.0, 1.0}})),
            std::invalid_argument);
        EXPECT_THROW(static_cast<void>(estimate_measurement_statistics({{0.0, 0.0}, {0.0, invalid}})),
            std::invalid_argument);
    }
}
