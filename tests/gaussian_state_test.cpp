#include "realtime_tracking/gaussian_state.hpp"
#include "realtime_tracking/process_noise.hpp"

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <limits>
#include <cstddef>
#include <stdexcept>

TEST(GaussianStatePrediction, PropagatesIdentityCovariance)
{
    const GaussianState current{
        Eigen::Vector4d{10.0, 5.0, 3.0, -2.0}, Eigen::Matrix4d::Identity()
    };
    const auto prediction = predict_gaussian_state(current, 1.0, 0.0);
    Eigen::Matrix4d expected;
    expected << 2.0, 0.0, 1.0, 0.0,
                0.0, 2.0, 0.0, 1.0,
                1.0, 0.0, 1.0, 0.0,
                0.0, 1.0, 0.0, 1.0;

    EXPECT_TRUE(prediction.mean.isApprox(Eigen::Vector4d{13.0, 3.0, 3.0, -2.0}, 1e-12));
    EXPECT_TRUE(prediction.covariance.isApprox(expected, 1e-12)) << prediction.covariance;
    EXPECT_TRUE(current.covariance.isIdentity());
}

TEST(GaussianStatePrediction, AddsProcessNoise)
{
    const GaussianState exact{Eigen::Vector4d::Zero(), Eigen::Matrix4d::Zero()};
    EXPECT_TRUE(predict_gaussian_state(exact, 2.0, 0.5).covariance.isApprox(
        constant_acceleration_process_noise(2.0, 0.5), 1e-12));
}

TEST(GaussianStatePrediction, RejectsAsymmetricCovariance)
{
    GaussianState invalid{Eigen::Vector4d::Zero(), Eigen::Matrix4d::Identity()};
    invalid.covariance(0, 1) = 0.25;
    EXPECT_THROW(static_cast<void>(predict_gaussian_state(invalid, 1.0, 0.0)), std::invalid_argument);
}

TEST(GaussianStatePrediction, RejectsNonfiniteMeanAndIndefiniteCovariance)
{
    GaussianState invalid{Eigen::Vector4d::Zero(), Eigen::Matrix4d::Identity()};
    invalid.mean(0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(static_cast<void>(predict_gaussian_state(invalid, 1.0, 0.0)), std::invalid_argument);
    invalid.mean.setZero();
    invalid.covariance(0, 0) = -1.0;
    EXPECT_THROW(static_cast<void>(predict_gaussian_state(invalid, 1.0, 0.0)), std::invalid_argument);
}

TEST(GaussianStatePrediction, RejectsNumericOverflow)
{
    const GaussianState large{
        Eigen::Vector4d::Constant(std::numeric_limits<double>::max()),
        Eigen::Matrix4d::Identity()
    };
    EXPECT_THROW(static_cast<void>(predict_gaussian_state(large, 2.0, 0.0)), std::overflow_error);
}

TEST(GaussianPredictor, CachedAndFreePredictionsAgree)
{
    const GaussianState current{
        Eigen::Vector4d{10.0, 5.0, 3.0, -2.0}, Eigen::Matrix4d::Identity()
    };
    const GaussianPredictor predictor{0.02, 0.5};
    const auto cached = predictor.predict(current);
    const auto uncached = predict_gaussian_state(current, 0.02, 0.5);
    EXPECT_TRUE(cached.mean.isApprox(uncached.mean, 1e-12));
    EXPECT_TRUE(cached.covariance.isApprox(uncached.covariance, 1e-12));
    EXPECT_TRUE(predictor.predict(current).covariance.isApprox(cached.covariance, 1e-12));
}

TEST(GaussianPredictor, RejectsInvalidConfiguration)
{
    EXPECT_THROW(static_cast<void>(GaussianPredictor(-1.0, 0.5)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(GaussianPredictor(0.0, 0.5)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(GaussianPredictor(1.0, -0.5)), std::invalid_argument);
}

TEST(GaussianPredictor, FourPredictionsEqualTwoSecondsOfMotion)
{
    GaussianState state{
        Eigen::Vector4d{1.0, 2.0, 3.0, -4.0}, Eigen::Matrix4d::Identity()
    };
    const GaussianPredictor predictor{0.5, 0.0};

    for (std::size_t step{}; step < 4; ++step)
    {
        if (step == 3)
        {
            EXPECT_NEAR(state.covariance(0, 2), 1.5, 1e-12);
        }
        state = predictor.predict(state);
    }

    Eigen::Matrix4d expected;
    expected << 5.0, 0.0, 2.0, 0.0,
                0.0, 5.0, 0.0, 2.0,
                2.0, 0.0, 1.0, 0.0,
                0.0, 2.0, 0.0, 1.0;
    EXPECT_TRUE(state.mean.isApprox(Eigen::Vector4d{7.0, -6.0, 3.0, -4.0}, 1e-12));
    EXPECT_TRUE(state.covariance.isApprox(expected, 1e-12)) << state.covariance;
}

TEST(GaussianStatePrediction, PropagatesContinuousProcessModel)
{
    const GaussianState posterior{
        Eigen::Vector4d{10.0, 20.0, 2.0, -1.0}, Eigen::Matrix4d::Identity()
    };
    const auto prediction = predict_gaussian_state(posterior, make_constant_velocity_model(1.0, 0.0));
    EXPECT_TRUE(prediction.mean.isApprox(Eigen::Vector4d{12.0, 19.0, 2.0, -1.0}, 1e-12));
    Eigen::Matrix4d expected;
    expected << 2.0, 0.0, 1.0, 0.0,
                0.0, 2.0, 0.0, 1.0,
                1.0, 0.0, 1.0, 0.0,
                0.0, 1.0, 0.0, 1.0;
    EXPECT_TRUE(prediction.covariance.isApprox(expected, 1e-12));
    const auto noisy_model = make_constant_velocity_model(2.0, 3.0);
    const GaussianState exact{Eigen::Vector4d::Zero(), Eigen::Matrix4d::Zero()};
    EXPECT_TRUE(predict_gaussian_state(exact, noisy_model).covariance.isApprox(
        noisy_model.process_covariance, 1e-12));
}

TEST(GaussianStatePrediction, RejectsInvalidProcessMatrices)
{
    const GaussianState posterior{Eigen::Vector4d::Zero(), Eigen::Matrix4d::Identity()};
    auto model = make_constant_velocity_model(1.0, 0.1);
    model.transition(0, 0) = std::numeric_limits<double>::infinity();
    EXPECT_THROW(static_cast<void>(predict_gaussian_state(posterior, model)), std::invalid_argument);
    model = make_constant_velocity_model(1.0, 0.1);
    model.process_covariance(0, 1) = 1.0;
    EXPECT_THROW(static_cast<void>(predict_gaussian_state(posterior, model)), std::invalid_argument);
    model.process_covariance = -Eigen::Matrix4d::Identity();
    EXPECT_THROW(static_cast<void>(predict_gaussian_state(posterior, model)), std::invalid_argument);
}

TEST(GaussianStatePrediction, RejectsIndefiniteCovarianceWithNonnegativeDiagonal)
{
    GaussianState posterior{Eigen::Vector4d::Zero(), Eigen::Matrix4d::Zero()};
    posterior.covariance(0, 1) = 1.0;
    posterior.covariance(1, 0) = 1.0;
    EXPECT_THROW(static_cast<void>(predict_gaussian_state(posterior, make_constant_velocity_model(1.0, 0.0))),
        std::invalid_argument);
}
