#include "realtime_tracking/image_noise.hpp"
#include "realtime_tracking/measurement_statistics.hpp"
#include "realtime_tracking/synthetic_target.hpp"
#include "realtime_tracking/target_tracker.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

TEST(TrackerSequence, LocalizesNoisyMovingTargets)
{
    TargetDetectorConfig configuration{};
    configuration.threshold = 125U;
    configuration.minimum_pixel_area = 20;
    configuration.opening_kernel_size = 3;
    configuration.maximum_bounding_box_aspect_ratio = 2.0;
    configuration.minimum_circularity = 0.5;
    TargetTracker tracker{configuration, {31, 31}};
    int successful{};
    int initial_detections{};
    int local_detections{};
    int reacquisitions{};
    int misses{};
    double sum_squared_error{};
    std::vector<Eigen::Vector2d> errors;

    for (int frame_index{}; frame_index < 20; ++frame_index)
    {
        const int true_x = 20 + 2 * frame_index;
        const int true_y = 30 + frame_index;
        const auto clean = make_synthetic_target_frame({100, 80, true_x, true_y, 5, 20U, 220U});
        const auto noisy = add_gaussian_sensor_noise(clean, 30.0,
            1000U + static_cast<std::uint64_t>(frame_index));
        const auto update = tracker.update(noisy);
        if (!update.has_value())
        {
            ++misses;
            continue;
        }

        ++successful;
        const double dx = update->detection.center_x - static_cast<double>(true_x);
        const double dy = update->detection.center_y - static_cast<double>(true_y);
        sum_squared_error += dx * dx + dy * dy;
        errors.emplace_back(dx, dy);
        switch (update->source)
        {
        case DetectionSource::InitialFullFrame:
            ++initial_detections;
            break;
        case DetectionSource::LocalSearch:
            ++local_detections;
            break;
        case DetectionSource::Reacquisition:
            ++reacquisitions;
            break;
        }
    }

    ASSERT_GT(successful, 0);
    const double success_rate = static_cast<double>(successful) / 20.0;
    const double rmse = std::sqrt(sum_squared_error / static_cast<double>(successful));
    EXPECT_GE(success_rate, 0.95);
    EXPECT_LT(rmse, 1.0);
    EXPECT_GT(local_detections, 0);
    EXPECT_EQ(initial_detections + local_detections + reacquisitions, successful);
    EXPECT_EQ(successful + misses, 20);
    RecordProperty("success_rate", std::to_string(success_rate));
    RecordProperty("position_rmse", std::to_string(rmse));
    RecordProperty("initial_detections", initial_detections);
    RecordProperty("local_detections", local_detections);
    RecordProperty("reacquisitions", reacquisitions);
    RecordProperty("misses", misses);
    ASSERT_GE(errors.size(), 2U);
    const auto statistics = estimate_measurement_statistics(errors);
    EXPECT_TRUE(statistics.mean_error.allFinite());
    EXPECT_TRUE(statistics.covariance.allFinite());
    EXPECT_TRUE(statistics.covariance.isApprox(statistics.covariance.transpose(), 1e-12));
    RecordProperty("mean_error_x", std::to_string(statistics.mean_error(0)));
    RecordProperty("mean_error_y", std::to_string(statistics.mean_error(1)));
    RecordProperty("stddev_x", std::to_string(std::sqrt(statistics.covariance(0, 0))));
    RecordProperty("stddev_y", std::to_string(std::sqrt(statistics.covariance(1, 1))));
}
