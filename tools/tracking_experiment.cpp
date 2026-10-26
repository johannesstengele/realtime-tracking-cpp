#include "realtime_tracking/kalman_target_tracker.hpp"
#include "realtime_tracking/kalman_reference.hpp"
#include "realtime_tracking/tracking_scenario.hpp"

#include <Eigen/Cholesky>
#include <opencv2/core.hpp>
#include <opencv2/core/version.hpp>

#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <stdexcept>
#include <string>
#include <string_view>

namespace
{
struct Options
{
    std::filesystem::path output;
    std::vector<TrackingScenario> scenarios{tracking_scenarios()};
    int runs{100};
    int frames{120};
    int warmup{10};
    std::uint64_t first_seed{10'000U};
    double measurement_variance{0.36};
    bool reference{false};
};

template <typename Number>
Number parse_number(std::string_view value)
{
    Number result{};
    const auto conversion = std::from_chars(value.data(), value.data() + value.size(), result);
    if (conversion.ec != std::errc{} || conversion.ptr != value.data() + value.size())
    {
        throw std::invalid_argument{"Invalid numeric option: " + std::string{value}};
    }
    return result;
}

Options parse_options(int argc, char** argv)
{
    Options options;
    for (int index{1}; index < argc; index += 2)
    {
        const std::string_view name{argv[index]};
        if (name == "--help")
        {
            std::cout << "tracking_experiment --output DIR [--scenario all|reference|NAME] [--runs 100] "
                         "[--frames 120] [--seed 10000] [--warmup 10] [--measurement-variance 0.36]\n";
            std::exit(0);
        }
        if (index + 1 >= argc)
        {
            throw std::invalid_argument{"Every option requires a value."};
        }
        const std::string_view value{argv[index + 1]};
        if (name == "--output") options.output = std::string{value};
        else if (name == "--scenario")
        {
            options.reference = value == "reference";
            options.scenarios = options.reference ? std::vector<TrackingScenario>{}
                : value == "all" ? tracking_scenarios()
                : std::vector<TrackingScenario>{parse_scenario(value)};
        }
        else if (name == "--runs") options.runs = parse_number<int>(value);
        else if (name == "--frames") options.frames = parse_number<int>(value);
        else if (name == "--seed") options.first_seed = parse_number<std::uint64_t>(value);
        else if (name == "--warmup") options.warmup = parse_number<int>(value);
        else if (name == "--measurement-variance") options.measurement_variance = parse_number<double>(value);
        else throw std::invalid_argument{"Unknown option: " + std::string{name}};
    }
    if (options.output.empty() || options.runs < 1 || options.runs > 10'000
        || options.frames < 1 || options.frames > 10'000
        || options.warmup < 0 || options.warmup >= options.frames
        || options.first_seed > std::numeric_limits<std::uint64_t>::max() - static_cast<std::uint64_t>(options.runs - 1)
        || !std::isfinite(options.measurement_variance) || options.measurement_variance <= 0.0)
    {
        throw std::invalid_argument{"Output directory, bounded positive runs/frames/variance and warmup < frames are required."};
    }
    return options;
}

std::string_view status_name(TrackStatus status)
{
    switch (status)
    {
    case TrackStatus::Searching: return "searching";
    case TrackStatus::Tracking: return "tracking";
    case TrackStatus::Coasting: return "coasting";
    case TrackStatus::Lost: return "lost";
    }
    throw std::logic_error{"Unknown track status."};
}

void run(const Options& options)
{
    std::filesystem::create_directories(options.output);
    if (options.reference)
    {
        const auto path = options.output / "reference.csv";
        if (std::filesystem::exists(path))
        {
            throw std::invalid_argument{"Reference output already exists; choose a new directory."};
        }
        const auto samples = run_kalman_reference(options.runs, options.frames, options.first_seed);
        std::ofstream reference{path};
        reference.exceptions(std::ios::badbit | std::ios::failbit);
        reference.imbue(std::locale::classic());
        reference << std::setprecision(17) << "seed,nis,nees\n";
        for (std::size_t index{}; index < samples.size(); ++index)
        {
            reference << options.first_seed + static_cast<std::uint64_t>(index) << ','
                << samples[index].nis << ',' << samples[index].nees << '\n';
        }
        reference.close();
        return;
    }
    const auto csv_path = options.output / "frames.csv";
    const auto metadata_path = options.output / "configuration.json";
    if (std::filesystem::exists(csv_path) || std::filesystem::exists(metadata_path))
    {
        throw std::invalid_argument{"Experiment output already exists; choose a new directory."};
    }
    std::ofstream csv{csv_path};
    csv.exceptions(std::ios::badbit | std::ios::failbit);
    csv.imbue(std::locale::classic());
    csv << std::setprecision(std::numeric_limits<double>::max_digits10);
    csv << "scenario,seed,frame,time,visible,true_x,true_y,true_vx,true_vy,status,track_id,misses,"
           "detected,measured_x,measured_y,estimated,estimate_x,estimate_y,estimate_vx,estimate_vy,"
           "position_squared_error,velocity_squared_error,measurement_squared_error,position_nees,nees,nis,"
           "render_ns,prediction_ns,detection_ns,association_ns,correction_ns,tracker_ns,pipeline_ns,timed\n";
    cv::setNumThreads(1);
    for (const auto scenario : options.scenarios)
    {
        for (int run_index{}; run_index < options.runs; ++run_index)
        {
            const auto seed = options.first_seed + static_cast<std::uint64_t>(run_index);
            const ScenarioConfig simulation{scenario, seed, options.frames, 1.0 / 30.0};
            KalmanTrackerConfig configuration;
            configuration.detector.threshold = 110U;
            configuration.detector.minimum_pixel_area = 20;
            configuration.detector.opening_kernel_size = 3;
            configuration.detector.smoothing_kernel_size = 3;
            configuration.detector.maximum_bounding_box_aspect_ratio = 2.0;
            configuration.detector.minimum_circularity = 0.45;
            configuration.measurement_covariance = Eigen::Matrix2d::Identity() * options.measurement_variance;
            configuration.collect_timings = true;
            KalmanTargetTracker tracker{configuration};
            for (int index{}; index < options.frames; ++index)
            {
                const auto start = std::chrono::steady_clock::now();
                const auto frame = render_tracking_frame(simulation, index);
                const auto rendered = std::chrono::steady_clock::now();
                const auto update = tracker.update(frame.image, simulation.delta_time);
                const auto stop = std::chrono::steady_clock::now();
                const auto& truth = frame.truth;
                csv << scenario_name(scenario) << ',' << seed << ',' << index << ','
                    << static_cast<double>(index) * simulation.delta_time << ',' << frame.visible << ','
                    << truth.x << ',' << truth.y << ',' << truth.vx << ',' << truth.vy << ','
                    << status_name(update.status) << ',' << update.track_id << ',' << update.consecutive_misses << ','
                    << update.detection.has_value() << ',';
                if (update.detection.has_value())
                {
                    csv << update.detection->center_x << ',' << update.detection->center_y;
                }
                else csv << ',';
                csv << ',' << update.state.has_value() << ',';
                if (update.state.has_value())
                {
                    const auto& state = *update.state;
                    const Eigen::Vector4d error = state.mean - Eigen::Vector4d{truth.x, truth.y, truth.vx, truth.vy};
                    const Eigen::Matrix2d position_covariance = state.covariance.topLeftCorner<2, 2>();
                    const Eigen::Vector2d position_error = error.head<2>();
                    const Eigen::LDLT<Eigen::Matrix2d> position_factor{position_covariance};
                    const Eigen::LDLT<Eigen::Matrix4d> state_factor{state.covariance};
                    if (position_factor.info() != Eigen::Success || state_factor.info() != Eigen::Success
                        || position_factor.vectorD().minCoeff() <= 0.0 || state_factor.vectorD().minCoeff() <= 0.0)
                    {
                        throw std::runtime_error{"Experiment covariance is not positive definite."};
                    }
                    csv << state.mean(0) << ',' << state.mean(1) << ',' << state.mean(2) << ',' << state.mean(3) << ','
                        << position_error.squaredNorm() << ',' << error.tail<2>().squaredNorm() << ',';
                    if (update.detection.has_value())
                    {
                        csv << std::pow(update.detection->center_x - truth.x, 2)
                            + std::pow(update.detection->center_y - truth.y, 2);
                    }
                    csv << ',' << position_error.dot(position_factor.solve(position_error))
                        << ',' << error.dot(state_factor.solve(error));
                }
                else csv << ",,,,,,,,";
                csv << ',';
                if (update.normalized_innovation_squared.has_value())
                {
                    csv << *update.normalized_innovation_squared;
                }
                const auto& timings = update.timings;
                csv << ',' << std::chrono::duration<double, std::nano>(rendered - start).count()
                    << ',' << timings.prediction_ns << ',' << timings.detection_ns << ',' << timings.association_ns
                    << ',' << timings.correction_ns << ',' << timings.total_ns << ','
                    << std::chrono::duration<double, std::nano>(stop - start).count()
                    << ',' << (index >= options.warmup) << '\n';
            }
        }
    }
    csv.close();
    std::ofstream metadata{metadata_path};
    metadata.exceptions(std::ios::badbit | std::ios::failbit);
    metadata.imbue(std::locale::classic());
    metadata << std::setprecision(17)
        << "{\n  \"schema_version\": 1,\n  \"width\": 320,\n  \"height\": 240,\n  \"fps\": 30,\n"
        << "  \"frames_per_run\": " << options.frames << ",\n  \"runs_per_scenario\": " << options.runs
        << ",\n  \"first_seed\": " << options.first_seed << ",\n  \"warmup_frames\": " << options.warmup
        << ",\n  \"measurement_variance\": " << options.measurement_variance
        << ",\n  \"acceleration_spectral_density\": 64,\n  \"initial_velocity_variance\": 400,\n"
        << "  \"gate_squared\": 9.210340371976184,\n  \"maximum_coast_frames\": 5,\n"
        << "  \"detector_threshold\": 110,\n  \"minimum_pixel_area\": 20,\n  \"opening_kernel_size\": 3,\n"
        << "  \"maximum_aspect_ratio\": 2,\n  \"minimum_circularity\": 0.45,\n  \"smoothing_kernel_size\": 3,\n  \"search_window\": [63, 63],\n"
        << "  \"camera\": {\"fx\": 200, \"fy\": 200, \"cx\": 160, \"cy\": 120, \"depth\": 5},\n"
        << "  \"normal_noise_sigma\": 30,\n  \"degraded_noise_sigma\": 35,\n  \"target_radius\": 5,\n"
        << "  \"short_occlusion_frames\": [45, 48],\n  \"long_dropout_frames\": [45, 57],\n"
        << "  \"opencv\": \"" << CV_VERSION << "\",\n  \"opencv_threads\": 1,\n  \"compiler\": \""
        << TRACKING_COMPILER << "\",\n  \"build_configuration\": \"" << TRACKING_BUILD_CONFIGURATION
        << "\",\n  \"scenarios\": [";
    bool first{true};
    for (const auto scenario : options.scenarios)
    {
        metadata << (first ? "" : ", ") << '"' << scenario_name(scenario) << '"';
        first = false;
    }
    metadata << "]\n}\n";
    metadata.close();
    std::cout << csv_path.string() << '\n';
}
}

int main(int argc, char** argv)
{
    try
    {
        run(parse_options(argc, argv));
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
