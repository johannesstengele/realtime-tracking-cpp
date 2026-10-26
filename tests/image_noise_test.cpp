#include "realtime_tracking/image_noise.hpp"
#include "realtime_tracking/synthetic_target.hpp"
#include "realtime_tracking/target_detection.hpp"

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

TEST(GaussianSensorNoise, IsReproducibleForFixedSeed)
{
    const cv::Mat clean{32, 32, CV_8UC1, cv::Scalar{128}};
    const auto original = clean.clone();
    const auto first = add_gaussian_sensor_noise(clean, 20.0, 12345U);
    const auto repeated = add_gaussian_sensor_noise(clean, 20.0, 12345U);
    const auto different = add_gaussian_sensor_noise(clean, 20.0, 54321U);
    EXPECT_EQ(first.type(), CV_8UC1);
    EXPECT_EQ(first.size(), clean.size());
    EXPECT_EQ(cv::countNonZero(first != repeated), 0);
    EXPECT_GT(cv::countNonZero(first != different), 0);
    EXPECT_EQ(cv::countNonZero(clean != original), 0);
}

TEST(GaussianSensorNoise, ZeroNoiseReturnsAnIndependentCopy)
{
    const cv::Mat clean{8, 8, CV_8UC1, cv::Scalar{128}};
    auto copy = add_gaussian_sensor_noise(clean, 0.0, 42U);
    EXPECT_EQ(cv::countNonZero(clean != copy), 0);
    copy.at<std::uint8_t>(0, 0) = 0U;
    EXPECT_EQ(clean.at<std::uint8_t>(0, 0), 128U);
}

TEST(GaussianSensorNoise, SaturatesExtremeNoiseWithoutIntegerWraparound)
{
    const cv::Mat clean{64, 64, CV_8UC1, cv::Scalar{128}};
    const auto noisy = add_gaussian_sensor_noise(clean, std::numeric_limits<double>::max(), 42U);
    EXPECT_GT(cv::countNonZero(noisy == 0), 0);
    EXPECT_GT(cv::countNonZero(noisy == 255), 0);
    EXPECT_EQ(cv::countNonZero((noisy != 0) & (noisy != 255)), 0);
}

TEST(GaussianSensorNoise, RejectsInvalidFrameAndDeviation)
{
    const cv::Mat clean{8, 8, CV_8UC1, cv::Scalar{128}};
    const cv::Mat color{8, 8, CV_8UC3, cv::Scalar{0, 0, 0}};
    EXPECT_THROW(static_cast<void>(add_gaussian_sensor_noise({}, 1.0, 1U)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(add_gaussian_sensor_noise(color, 1.0, 1U)), std::invalid_argument);
    for (const double invalid : {-1.0, std::numeric_limits<double>::infinity(),
                                 std::numeric_limits<double>::quiet_NaN()})
    {
        EXPECT_THROW(static_cast<void>(add_gaussian_sensor_noise(clean, invalid, 1U)), std::invalid_argument);
    }
}

TEST(GaussianSensorNoise, DetectorLocalizesNoisyTarget)
{
    const auto clean = make_synthetic_target_frame({96, 72, 48, 36, 8, 30U, 220U});
    const auto noisy = add_gaussian_sensor_noise(clean, 35.0, 20260916U);
    TargetDetectorConfig configuration{};
    configuration.threshold = 125U;
    configuration.minimum_pixel_area = 50;
    configuration.opening_kernel_size = 3;
    configuration.maximum_bounding_box_aspect_ratio = 2.0;
    configuration.minimum_circularity = 0.5;
    const auto detection = detect_bright_target(noisy, configuration);
    ASSERT_TRUE(detection.has_value());
    EXPECT_LE(std::hypot(detection->center_x - 48.0, detection->center_y - 36.0), 0.75);
}
