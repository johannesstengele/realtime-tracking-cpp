#pragma once

#include "realtime_tracking/state2d.hpp"

#include <filesystem>
#include <vector>

void write_trajectory_csv(
    const std::vector<State2D>& trajectory,
    const std::filesystem::path& output_path);