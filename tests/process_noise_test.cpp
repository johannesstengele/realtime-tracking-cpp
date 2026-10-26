#include "realtime_tracking/process_noise.hpp"

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace
{
using NoiseCase = std::tuple<double, double>;

class ProcessNoiseInvariantTest : public ::testing::TestWithParam<NoiseCase>
{
};
}

TEST_P(ProcessNoiseInvariantTest, ProducesValidCovariance)
{
    const auto [delta_time, acceleration_sigma] = GetParam();
    const auto covariance = constant_acceleration_process_noise(delta_time, acceleration_sigma);
    EXPECT_TRUE(covariance.allFinite());
    EXPECT_TRUE(covariance.isApprox(covariance.transpose(), 1e-12));
    EXPECT_GE(covariance.diagonal().minCoeff(), 0.0);
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> solver{covariance};
    ASSERT_EQ(solver.info(), Eigen::Success);
    const double scale = std::max(1.0, covariance.cwiseAbs().maxCoeff());
    EXPECT_GE(solver.eigenvalues().minCoeff(), -1e-12 * scale);
}

INSTANTIATE_TEST_SUITE_P(RepresentativeScales, ProcessNoiseInvariantTest,
    ::testing::Values(NoiseCase{0.001, 0.0}, NoiseCase{0.02, 0.5},
                      NoiseCase{1.0, 2.0}, NoiseCase{10.0, 100.0}));

TEST(ProcessNoise, VarianceScalesWithSquaredAccelerationDeviation)
{
    EXPECT_TRUE(constant_acceleration_process_noise(0.3, 1.5).isApprox(
        9.0 * constant_acceleration_process_noise(0.3, 0.5), 1e-12));
}

TEST(ProcessNoise, RejectsZeroTimeStepAndOverflow)
{
    EXPECT_THROW(static_cast<void>(constant_acceleration_process_noise(0.0, 0.5)),
        std::invalid_argument);
    EXPECT_THROW(static_cast<void>(constant_acceleration_process_noise(
        std::numeric_limits<double>::max(), 1.0)), std::overflow_error);
    EXPECT_TRUE(constant_acceleration_process_noise(
        std::numeric_limits<double>::max(), 0.0).isZero());
}
