#include "realtime_tracking/kalman_reference.hpp"

#include "realtime_tracking/gaussian_state.hpp"
#include "realtime_tracking/position_measurement.hpp"

#include <Eigen/Cholesky>

#include <limits>
#include <random>
#include <stdexcept>

std::vector<ConsistencySample> run_kalman_reference(int trials, int steps, std::uint64_t first_seed)
{
    if (trials <= 0 || trials > 100'000 || steps <= 0 || steps > 10'000
        || first_seed > std::numeric_limits<std::uint64_t>::max() - static_cast<std::uint64_t>(trials - 1))
    {
        throw std::invalid_argument{"Reference trial count, step count or seed range is invalid."};
    }
    const auto model = make_constant_velocity_model(1.0 / 30.0, 64.0);
    const Eigen::Matrix4d process_root = model.process_covariance.llt().matrixL();
    const Eigen::Matrix2d measurement_covariance = Eigen::Matrix2d::Identity() * 0.36;
    std::vector<ConsistencySample> samples;
    samples.reserve(static_cast<std::size_t>(trials));
    for (int trial{}; trial < trials; ++trial)
    {
        std::mt19937_64 generator{first_seed + static_cast<std::uint64_t>(trial)};
        std::normal_distribution<double> normal;
        GaussianState posterior{Eigen::Vector4d::Zero(), Eigen::Matrix4d::Identity()};
        Eigen::Vector4d truth;
        for (Eigen::Index index{}; index < 4; ++index) truth(index) = normal(generator);
        ConsistencySample sample{};
        for (int step{}; step < steps; ++step)
        {
            Eigen::Vector4d driving_noise;
            for (Eigen::Index index{}; index < 4; ++index) driving_noise(index) = normal(generator);
            truth = model.transition * truth + process_root * driving_noise;
            const Eigen::Vector2d measurement = truth.head<2>()
                + 0.6 * Eigen::Vector2d{normal(generator), normal(generator)};
            const auto prior = predict_gaussian_state(posterior, model);
            const auto innovation = compute_position_innovation(prior, measurement, measurement_covariance);
            sample.nis = innovation.innovation.dot(
                innovation.innovation_covariance.ldlt().solve(innovation.innovation));
            posterior = correct_position_measurement(prior, measurement, measurement_covariance);
            const Eigen::Vector4d error = posterior.mean - truth;
            sample.nees = error.dot(posterior.covariance.ldlt().solve(error));
        }
        samples.push_back(sample);
    }
    return samples;
}
