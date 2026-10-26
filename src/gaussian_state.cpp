#include "realtime_tracking/gaussian_state.hpp"

#include "realtime_tracking/process_noise.hpp"
#include "realtime_tracking/state_transition.hpp"
#include "covariance_validation.hpp"

#include <stdexcept>

namespace
{
GaussianState predict_with_matrices(
    const GaussianState& current,
    const Eigen::Matrix4d& transition,
    const Eigen::Matrix4d& process_noise)
{
    if (!current.mean.allFinite())
    {
        throw std::invalid_argument{"State mean must be finite."};
    }

    tracking_detail::validate_covariance(current.covariance);
    GaussianState prediction{
        transition * current.mean,
        transition * current.covariance * transition.transpose() + process_noise
    };
    prediction.covariance = (0.5 * prediction.covariance
        + 0.5 * prediction.covariance.transpose()).eval();

    if (!prediction.mean.allFinite() || !prediction.covariance.allFinite())
    {
        throw std::overflow_error{"Predicted state exceeds the finite numeric range."};
    }

    return prediction;
}
}

GaussianPredictor::GaussianPredictor(double delta_time, double acceleration_sigma)
    : transition_{constant_velocity_transition(delta_time)},
      process_noise_{constant_acceleration_process_noise(delta_time, acceleration_sigma)}
{
}

GaussianState GaussianPredictor::predict(const GaussianState& current) const
{
    return predict_with_matrices(current, transition_, process_noise_);
}

GaussianState predict_gaussian_state(
    const GaussianState& current,
    double delta_time,
    double acceleration_sigma)
{
    return GaussianPredictor{delta_time, acceleration_sigma}.predict(current);
}

GaussianState predict_gaussian_state(
    const GaussianState& posterior,
    const ConstantVelocityProcessModel& model)
{
    if (!model.transition.allFinite())
    {
        throw std::invalid_argument{"State transition must be finite."};
    }
    tracking_detail::validate_covariance(model.process_covariance);
    return predict_with_matrices(posterior, model.transition, model.process_covariance);
}
