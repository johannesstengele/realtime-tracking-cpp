#include "realtime_tracking/process_noise.hpp"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <iostream>
#include <limits>
#include <stdexcept>

int main()
{
    int failures{};
    Eigen::Matrix4d expected;
    expected << 1.0, 0.0, 1.0, 0.0,
                0.0, 1.0, 0.0, 1.0,
                1.0, 0.0, 1.0, 0.0,
                0.0, 1.0, 0.0, 1.0;
    const auto covariance = constant_acceleration_process_noise(2.0, 0.5);
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> solver{covariance};

    if (!covariance.isApprox(expected, 1e-12)
        || !covariance.isApprox(covariance.transpose(), 1e-12)
        || solver.info() != Eigen::Success || solver.eigenvalues().minCoeff() < -1e-12
        || !constant_acceleration_process_noise(2.0, 0.0).isZero())
    {
        std::cerr << "Incorrect acceleration process covariance.\n";
        ++failures;
    }

    for (const double invalid : {-1.0, std::numeric_limits<double>::infinity(),
                                 std::numeric_limits<double>::quiet_NaN()})
    {
        for (const bool invalid_time : {false, true})
        {
            bool rejected{false};
            try
            {
                static_cast<void>(constant_acceleration_process_noise(
                    invalid_time ? invalid : 1.0, invalid_time ? 1.0 : invalid));
            }
            catch (const std::invalid_argument&)
            {
                rejected = true;
            }
            if (!rejected)
            {
                std::cerr << "Accepted invalid process-noise parameter.\n";
                ++failures;
            }
        }
    }
    return (failures == 0) ? 0 : 1;
}
