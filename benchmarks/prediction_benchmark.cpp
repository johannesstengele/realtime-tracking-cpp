#include "realtime_tracking/gaussian_state.hpp"

#include <Eigen/Core>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>

int main()
{
    constexpr std::size_t warmup_iterations{1'000};
    constexpr std::size_t measured_iterations{100'000};
    GaussianState current{
        Eigen::Vector4d{10.0, 5.0, 3.0, -2.0}, Eigen::Matrix4d::Identity()
    };
    const GaussianPredictor predictor{0.02, 0.5};
    double warmup_checksum{};

    for (std::size_t index{}; index < warmup_iterations; ++index)
    {
        current.mean(0) = static_cast<double>(index) * 1e-6;
        const auto result = predictor.predict(current);
        warmup_checksum += result.mean(0) + result.covariance(0, 0);
    }

    double checksum{};
    const auto start = std::chrono::steady_clock::now();

    for (std::size_t index{}; index < measured_iterations; ++index)
    {
        current.mean(0) = static_cast<double>(index) * 1e-6;
        const auto result = predictor.predict(current);
        checksum += result.mean(0) + result.covariance(0, 0);
    }

    const auto stop = std::chrono::steady_clock::now();
    const double elapsed_nanoseconds =
        std::chrono::duration<double, std::nano>{stop - start}.count();

    std::cout << std::fixed << std::setprecision(2)
              << "iterations=" << measured_iterations << '\n'
              << "total_ms=" << elapsed_nanoseconds / 1'000'000.0 << '\n'
              << "mean_ns=" << elapsed_nanoseconds / static_cast<double>(measured_iterations) << '\n'
              << "warmup_checksum=" << warmup_checksum << '\n'
              << "checksum=" << checksum << '\n';

    return (std::isfinite(checksum) && std::isfinite(elapsed_nanoseconds)) ? 0 : 1;
}
