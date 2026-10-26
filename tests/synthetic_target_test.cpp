#include "realtime_tracking/synthetic_target.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

TEST(SyntheticTargetFrame, CreatesExpectedGrayscaleGeometry)
{
    const auto frame = make_synthetic_target_frame({64, 48, 30, 20, 5, 10U, 240U});
    EXPECT_EQ(frame.cols, 64);
    EXPECT_EQ(frame.rows, 48);
    EXPECT_EQ(frame.type(), CV_8UC1);
    EXPECT_EQ(frame.at<std::uint8_t>(20, 30), 240U);
    EXPECT_EQ(frame.at<std::uint8_t>(0, 0), 10U);
}

TEST(SyntheticTargetFrame, RejectsInvalidGeometryWithoutIntegerOverflow)
{
    const std::array<SyntheticTargetSpec, 8> cases{{
        {0, 48, 30, 20, 5, 0U, 255U},
        {64, -1, 30, 20, 5, 0U, 255U},
        {64, 48, -1, 20, 5, 0U, 255U},
        {64, 48, 30, 48, 5, 0U, 255U},
        {64, 48, 30, 20, 0, 0U, 255U},
        {64, 48, 2, 20, 5, 0U, 255U},
        {64, 48, 60, 20, 5, 0U, 255U},
        {64, 48, 30, 20, std::numeric_limits<int>::max(), 0U, 255U}
    }};
    for (const auto& specification : cases)
    {
        EXPECT_THROW(static_cast<void>(make_synthetic_target_frame(specification)),
            std::invalid_argument);
    }
}

TEST(SyntheticTargetFrame, PermitsDiskTouchingFinalValidPixels)
{
    const auto frame = make_synthetic_target_frame({11, 11, 5, 5, 5, 0U, 255U});
    EXPECT_EQ(frame.at<std::uint8_t>(5, 0), 255U);
    EXPECT_EQ(frame.at<std::uint8_t>(5, 10), 255U);
    EXPECT_EQ(frame.at<std::uint8_t>(0, 5), 255U);
    EXPECT_EQ(frame.at<std::uint8_t>(10, 5), 255U);
}
