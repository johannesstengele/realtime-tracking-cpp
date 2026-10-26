#pragma once

#include <Eigen/Core>

[[nodiscard]] Eigen::Matrix4d constant_velocity_transition(double delta_time);
