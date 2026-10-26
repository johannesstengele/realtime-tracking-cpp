#pragma once

#include <Eigen/Core>

[[nodiscard]] Eigen::Matrix4d constant_acceleration_process_noise(
    double delta_time,
    double acceleration_sigma);
