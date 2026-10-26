#include "realtime_tracking/kalman_reference.hpp"

#include <gtest/gtest.h>

#include <numeric>
#include <stdexcept>

TEST(KalmanReference, IndependentFinalEpochSamplesHaveConsistentUncertainty)
{
    const auto samples = run_kalman_reference(1000, 20, 800'000U);
    const double nis = std::accumulate(samples.begin(), samples.end(), 0.0,
        [](double total, const ConsistencySample& sample) { return total + sample.nis; }) / 1000.0;
    const double nees = std::accumulate(samples.begin(), samples.end(), 0.0,
        [](double total, const ConsistencySample& sample) { return total + sample.nees; }) / 1000.0;
    EXPECT_GT(nis, 1.75);
    EXPECT_LT(nis, 2.25);
    EXPECT_GT(nees, 3.6);
    EXPECT_LT(nees, 4.4);
    RecordProperty("mean_nis", nis);
    RecordProperty("mean_nees", nees);
}

TEST(KalmanReference, RejectsUnboundedOrEmptyExperiments)
{
    EXPECT_THROW(static_cast<void>(run_kalman_reference(0, 20, 1U)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(run_kalman_reference(10, 0, 1U)), std::invalid_argument);
}
