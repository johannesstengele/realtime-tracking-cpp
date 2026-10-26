#pragma once

#include <cstdint>
#include <vector>

struct ConsistencySample
{
    double nis;
    double nees;
};

[[nodiscard]] std::vector<ConsistencySample> run_kalman_reference(
    int trials, int steps, std::uint64_t first_seed);
