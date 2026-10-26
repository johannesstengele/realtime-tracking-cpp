#include "realtime_tracking/position_measurement.hpp"
#include "realtime_tracking/trajectory_analysis.hpp"
#include "realtime_tracking/trajectory_simulator.hpp"

#include <gtest/gtest.h>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <limits>
#include <stdexcept>
#include <vector>

TEST(PositionMeasurement, ComputesInnovationAndCovariance)
{
    GaussianState prediction{Eigen::Vector4d{10.0, 20.0, 2.0, -1.0}, Eigen::Matrix4d::Zero()};
    prediction.covariance.diagonal() << 4.0, 9.0, 1.0, 1.0;
    Eigen::Matrix2d measurement_covariance = Eigen::Matrix2d::Zero();
    measurement_covariance.diagonal() << 1.0, 4.0;
    const auto result = compute_position_innovation(prediction, {12.0, 17.0}, measurement_covariance);
    EXPECT_TRUE(result.predicted_measurement.isApprox(Eigen::Vector2d{10.0, 20.0}, 1e-12));
    EXPECT_TRUE(result.innovation.isApprox(Eigen::Vector2d{2.0, -3.0}, 1e-12));
    Eigen::Matrix2d expected;
    expected << 5.0, 0.0, 0.0, 13.0;
    EXPECT_TRUE(result.innovation_covariance.isApprox(expected, 1e-12));
    const double normalized_innovation = result.innovation.dot(
        result.innovation_covariance.ldlt().solve(result.innovation));
    EXPECT_NEAR(normalized_innovation, 4.0 / 5.0 + 9.0 / 13.0, 1e-12);
}

TEST(PositionMeasurement, PreservesCorrelatedMeasurementUncertainty)
{
    GaussianState prediction{Eigen::Vector4d::Zero(), Eigen::Matrix4d::Identity()};
    prediction.covariance(0, 0) = 4.0;
    prediction.covariance(1, 1) = 9.0;
    prediction.covariance(0, 1) = 1.0;
    prediction.covariance(1, 0) = 1.0;
    Eigen::Matrix2d noise;
    noise << 2.0, 0.5, 0.5, 3.0;
    const auto result = compute_position_innovation(prediction, {1.0, 2.0}, noise);
    Eigen::Matrix2d expected;
    expected << 6.0, 1.5, 1.5, 12.0;
    EXPECT_TRUE(result.innovation_covariance.isApprox(expected, 1e-12));
}

TEST(PositionMeasurement, RejectsInvalidCovarianceAcrossScales)
{
    const GaussianState prediction{Eigen::Vector4d::Zero(), Eigen::Matrix4d::Identity()};
    Eigen::Matrix2d covariance;
    covariance << 1.0, 2.0, 2.0, 1.0;
    EXPECT_THROW(static_cast<void>(compute_position_innovation(prediction, {0.0, 0.0}, covariance)),
        std::invalid_argument);
    covariance << 1e-30, 2e-15, 2e-15, 1.0;
    EXPECT_THROW(static_cast<void>(compute_position_innovation(prediction, {0.0, 0.0}, covariance)),
        std::invalid_argument);
    covariance << 1.0, 0.5, 0.0, 1.0;
    EXPECT_THROW(static_cast<void>(compute_position_innovation(prediction, {0.0, 0.0}, covariance)),
        std::invalid_argument);
}

TEST(PositionMeasurement, RejectsNonfiniteMeasurementsAndOverflow)
{
    GaussianState prediction{Eigen::Vector4d::Zero(), Eigen::Matrix4d::Identity()};
    EXPECT_THROW(static_cast<void>(compute_position_innovation(prediction,
        {std::numeric_limits<double>::quiet_NaN(), 0.0}, Eigen::Matrix2d::Identity())), std::invalid_argument);
    prediction.mean(0) = std::numeric_limits<double>::max();
    EXPECT_THROW(static_cast<void>(compute_position_innovation(prediction,
        {-std::numeric_limits<double>::max(), 0.0}, Eigen::Matrix2d::Identity())), std::overflow_error);
}

TEST(KalmanCorrection, MatchesExactMeanAndJosephCovariance)
{
    GaussianState prediction{Eigen::Vector4d{10.0, 20.0, 0.0, 0.0}, Eigen::Matrix4d::Zero()};
    prediction.covariance.diagonal() << 4.0, 9.0, 1.0, 1.0;
    Eigen::Matrix2d noise = Eigen::Matrix2d::Zero();
    noise.diagonal() << 1.0, 4.0;
    const auto corrected = correct_position_measurement(prediction, {12.0, 17.0}, noise);
    EXPECT_TRUE(corrected.mean.isApprox(Eigen::Vector4d{11.6, 20.0 - 27.0 / 13.0, 0.0, 0.0}, 1e-12));
    Eigen::Matrix4d expected = Eigen::Matrix4d::Zero();
    expected.diagonal() << 0.8, 36.0 / 13.0, 1.0, 1.0;
    EXPECT_TRUE(corrected.covariance.isApprox(expected, 1e-12));
    EXPECT_TRUE(corrected.covariance.isApprox(corrected.covariance.transpose(), 1e-12));
    EXPECT_DOUBLE_EQ(prediction.mean(0), 10.0);
    EXPECT_DOUBLE_EQ(prediction.covariance(0, 0), 4.0);
}

TEST(KalmanCorrection, PositionMeasurementsCorrectCorrelatedVelocity)
{
    GaussianState prediction{Eigen::Vector4d{10.0, 20.0, 0.0, 0.0}, Eigen::Matrix4d::Zero()};
    prediction.covariance.diagonal() << 4.0, 9.0, 1.0, 1.0;
    prediction.covariance(0, 2) = prediction.covariance(2, 0) = 1.0;
    prediction.covariance(1, 3) = prediction.covariance(3, 1) = 2.0;
    Eigen::Matrix2d noise = Eigen::Matrix2d::Zero();
    noise.diagonal() << 1.0, 4.0;
    const auto corrected = correct_position_measurement(prediction, {12.0, 17.0}, noise);
    EXPECT_NEAR(corrected.mean(2), 0.4, 1e-12);
    EXPECT_NEAR(corrected.mean(3), -6.0 / 13.0, 1e-12);
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> solver{corrected.covariance};
    ASSERT_EQ(solver.info(), Eigen::Success);
    EXPECT_GE(solver.eigenvalues().minCoeff(), -1e-12);
}

TEST(KalmanCorrection, HandlesPerfectMeasurementsAndRejectsSingularInnovation)
{
    GaussianState prediction{Eigen::Vector4d::Zero(), Eigen::Matrix4d::Identity()};
    const auto exact = correct_position_measurement(prediction, {3.0, -2.0}, Eigen::Matrix2d::Zero());
    EXPECT_DOUBLE_EQ(exact.mean(0), 3.0);
    EXPECT_DOUBLE_EQ(exact.mean(1), -2.0);
    EXPECT_TRUE((exact.covariance.topLeftCorner<2, 2>().isZero()));
    prediction.covariance(0, 0) = 0.0;
    EXPECT_THROW(static_cast<void>(correct_position_measurement(prediction,
        {0.0, 0.0}, Eigen::Matrix2d::Zero())), std::runtime_error);
    prediction.covariance.setZero();
    EXPECT_THROW(static_cast<void>(correct_position_measurement(prediction,
        {0.0, 0.0}, Eigen::Matrix2d::Zero())), std::runtime_error);
}

TEST(KalmanCorrection, RejectsNonfiniteMeasurementCovariance)
{
    const GaussianState prediction{Eigen::Vector4d::Zero(), Eigen::Matrix4d::Identity()};
    Eigen::Matrix2d noise = Eigen::Matrix2d::Identity();
    noise(1, 1) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(static_cast<void>(correct_position_measurement(prediction, {0.0, 0.0}, noise)),
        std::invalid_argument);
}

TEST(KalmanCorrection, ReducesErrorOverRepeatedPredictionAndCorrection)
{
    const auto truth = simulate_trajectory(ConstantVelocityModel{1.0}, {0.0, 0.0, 2.0, -1.0}, 100);
    const auto measurements = add_position_noise(truth, 2.0, 1337U);
    const auto model = make_constant_velocity_model(1.0, 0.01);
    GaussianState posterior{Eigen::Vector4d::Zero(), Eigen::Matrix4d::Identity() * 10.0};
    const Eigen::Matrix2d noise = Eigen::Matrix2d::Identity() * 4.0;
    std::vector<State2D> estimates;
    estimates.reserve(truth.size());
    for (std::size_t index{}; index < truth.size(); ++index)
    {
        if (index > 0)
        {
            posterior = predict_gaussian_state(posterior, model);
        }
        posterior = correct_position_measurement(posterior,
            {measurements[index].x, measurements[index].y}, noise);
        EXPECT_TRUE(posterior.mean.allFinite());
        EXPECT_TRUE(posterior.covariance.isApprox(posterior.covariance.transpose(), 1e-12));
        estimates.push_back({posterior.mean(0), posterior.mean(1), posterior.mean(2), posterior.mean(3)});
    }
    EXPECT_LT(position_rmse(truth, estimates), position_rmse(truth, measurements));
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> solver{posterior.covariance};
    ASSERT_EQ(solver.info(), Eigen::Success);
    EXPECT_GE(solver.eigenvalues().minCoeff(), -1e-12);
    EXPECT_NEAR(posterior.mean(2), 2.0, 1.0);
    EXPECT_NEAR(posterior.mean(3), -1.0, 1.0);
}
