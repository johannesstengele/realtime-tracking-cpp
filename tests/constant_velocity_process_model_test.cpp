#include "realtime_tracking/constant_velocity_model.hpp"

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <limits>
#include <stdexcept>

TEST(ConstantVelocityProcessModel, ConstructsTransitionAndContinuousAccelerationCovariance)
{
    const auto model = make_constant_velocity_model(2.0, 3.0);
    const Eigen::Vector4d prediction = model.transition * Eigen::Vector4d{10.0, 20.0, 2.0, -1.0};
    EXPECT_TRUE(prediction.isApprox(Eigen::Vector4d{14.0, 18.0, 2.0, -1.0}, 1e-12));
    Eigen::Matrix4d expected;
    expected << 8.0, 0.0, 6.0, 0.0,
                0.0, 8.0, 0.0, 6.0,
                6.0, 0.0, 6.0, 0.0,
                0.0, 6.0, 0.0, 6.0;
    EXPECT_TRUE(model.process_covariance.isApprox(expected, 1e-12));
    EXPECT_TRUE(model.process_covariance.isApprox(model.process_covariance.transpose(), 1e-12));
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> solver{model.process_covariance};
    ASSERT_EQ(solver.info(), Eigen::Success);
    EXPECT_GT(solver.eigenvalues().minCoeff(), 0.0);
}

TEST(ConstantVelocityProcessModel, IndependentIntervalsComposeConsistently)
{
    const auto first = make_constant_velocity_model(0.3, 0.2);
    const auto second = make_constant_velocity_model(0.7, 0.2);
    const auto combined = make_constant_velocity_model(1.0, 0.2);
    EXPECT_TRUE((second.transition * first.process_covariance * second.transition.transpose()
        + second.process_covariance).isApprox(combined.process_covariance, 1e-12));
    EXPECT_TRUE(make_constant_velocity_model(1.0, 0.0).process_covariance.isZero());
}

TEST(ConstantVelocityProcessModel, RejectsInvalidParametersAndOverflow)
{
    for (const double dt : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
                            std::numeric_limits<double>::infinity()})
    {
        EXPECT_THROW(static_cast<void>(make_constant_velocity_model(dt, 1.0)), std::invalid_argument);
    }
    for (const double q : {-1.0, std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity()})
    {
        EXPECT_THROW(static_cast<void>(make_constant_velocity_model(1.0, q)), std::invalid_argument);
    }
    EXPECT_THROW(static_cast<void>(make_constant_velocity_model(std::numeric_limits<double>::max(), 1.0)),
        std::overflow_error);
}

TEST(ConstantVelocityProcessModel, DeterministicModelAcceptsEveryFinitePositiveTimeStep)
{
    const ConstantVelocityModel model{1e-20};
    EXPECT_DOUBLE_EQ(model.propagate({0.0, 0.0, 1.0, 0.0}).x, 1e-20);
}
