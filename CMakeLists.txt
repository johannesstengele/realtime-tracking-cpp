cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(realtime_tracking VERSION 0.2.0 LANGUAGES CXX)

include(FetchContent)
include(cmake/ProjectOptions.cmake)

function(tracking_fetch_eigen)
    set(BUILD_TESTING OFF)
    set(EIGEN_BUILD_DOC OFF)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    set(CMAKE_EXPORT_NO_PACKAGE_REGISTRY ON)
    FetchContent_Declare(eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(eigen)
    target_include_directories(eigen SYSTEM INTERFACE "$<BUILD_INTERFACE:${eigen_SOURCE_DIR}>")
endfunction()

find_package(Eigen3 3.4 CONFIG QUIET)
if(NOT TARGET Eigen3::Eigen)
    tracking_fetch_eigen()
endif()

add_library(tracking
    src/constant_velocity_model.cpp
    src/trajectory_simulator.cpp
    src/trajectory_analysis.cpp
    src/trajectory_io.cpp
    src/state_transition.cpp
    src/process_noise.cpp
    src/gaussian_state.cpp
    src/measurement_statistics.cpp
    src/position_measurement.cpp
    src/demonstrate_rejection.cpp
    src/kalman_reference.cpp
)

target_include_directories(tracking
    PUBLIC
        include
)

target_compile_features(tracking PUBLIC cxx_std_20)
target_link_libraries(tracking PUBLIC Eigen3::Eigen)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc)

foreach(tracking_opencv_target IN ITEMS opencv_core opencv_imgproc)
    get_target_property(tracking_opencv_configurations ${tracking_opencv_target} IMPORTED_CONFIGURATIONS)
    list(TRANSFORM tracking_opencv_configurations TOUPPER)
    if("RELWITHDEBINFO" IN_LIST tracking_opencv_configurations
        AND NOT "RELEASE" IN_LIST tracking_opencv_configurations)
        set_target_properties(${tracking_opencv_target} PROPERTIES
            MAP_IMPORTED_CONFIG_RELWITHDEBINFO RelWithDebInfo
            MAP_IMPORTED_CONFIG_MINSIZEREL RelWithDebInfo
        )
    endif()
endforeach()

add_library(vision
    src/synthetic_target.cpp
    src/target_detection.cpp
    src/image_noise.cpp
    src/search_region.cpp
    src/target_tracker.cpp
    src/kalman_target_tracker.cpp
    src/tracking_scenario.cpp
)
target_include_directories(vision PUBLIC include)
target_include_directories(vision SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
target_compile_features(vision PUBLIC cxx_std_20)
target_link_libraries(vision PUBLIC opencv_core opencv_imgproc tracking)

add_executable(realtime_tracking
    src/main.cpp
)

target_link_libraries(realtime_tracking
    PRIVATE
        tracking
        vision
)

add_executable(tracking_experiment tools/tracking_experiment.cpp)
target_link_libraries(tracking_experiment PRIVATE vision)
target_compile_definitions(tracking_experiment PRIVATE
    TRACKING_COMPILER="${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}"
    TRACKING_BUILD_CONFIGURATION="$<CONFIG>"
)

include(CTest)

if(BUILD_TESTING)
    find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)
    add_test(NAME evaluation_interface
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_evaluation.py
            $<TARGET_FILE:tracking_experiment>
    )
    set_tests_properties(evaluation_interface PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}"
    )
    add_test(NAME tracking_example COMMAND realtime_tracking)

    add_executable(state_transition_tests tests/test_state_transition.cpp)
    target_link_libraries(state_transition_tests PRIVATE tracking)
    add_test(NAME state_transition COMMAND state_transition_tests)

    add_executable(process_noise_tests tests/test_process_noise.cpp)
    target_link_libraries(process_noise_tests PRIVATE tracking)
    add_test(NAME process_noise COMMAND process_noise_tests)

    find_package(GTest 1.18 CONFIG QUIET)
    if(NOT TARGET GTest::gtest_main)
        set(gtest_force_shared_crt ON CACHE BOOL "Use the shared MSVC runtime")
        set(INSTALL_GTEST OFF CACHE BOOL "Disable GoogleTest installation")
        set(BUILD_GMOCK OFF CACHE BOOL "Disable GoogleMock")
        FetchContent_Declare(googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.18.0
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(googletest)
    endif()
    if(TRACKING_ENABLE_SANITIZERS AND TARGET gtest AND TARGET gtest_main)
        target_link_libraries(gtest PRIVATE project_sanitizers)
        target_link_libraries(gtest_main PRIVATE project_sanitizers)
    endif()
    include(GoogleTest)

    add_executable(gaussian_state_tests
        tests/gaussian_state_test.cpp
        tests/process_noise_test.cpp
        tests/measurement_statistics_test.cpp
        tests/constant_velocity_process_model_test.cpp
        tests/position_measurement_test.cpp
        tests/kalman_reference_test.cpp
    )
    target_link_libraries(gaussian_state_tests PRIVATE tracking GTest::gtest_main)
    gtest_discover_tests(gaussian_state_tests DISCOVERY_MODE PRE_TEST)

    add_executable(vision_tests
        tests/synthetic_target_test.cpp
        tests/target_detection_test.cpp
        tests/image_noise_test.cpp
        tests/search_region_test.cpp
        tests/target_tracker_test.cpp
        tests/tracker_sequence_test.cpp
        tests/kalman_target_tracker_test.cpp
        tests/tracking_scenario_test.cpp
        tests/tracking_pipeline_test.cpp
    )
    target_link_libraries(vision_tests PRIVATE vision tracking GTest::gtest_main)
    gtest_discover_tests(vision_tests DISCOVERY_MODE PRE_TEST)

    add_executable(tracking_tests
        tests/test_constant_velocity_model.cpp
    )

    target_link_libraries(tracking_tests
        PRIVATE
            tracking
    )

    add_test(
        NAME constant_velocity_model
        COMMAND tracking_tests
    )

    add_executable(simulation_tests
        tests/test_trajectory_simulator.cpp
    )

    target_link_libraries(simulation_tests
        PRIVATE
            tracking
    )

    add_test(
        NAME trajectory_simulator
        COMMAND simulation_tests
    )

    add_executable(trajectory_io_tests
        tests/test_trajectory_io.cpp
    )

    target_link_libraries(trajectory_io_tests
        PRIVATE
            tracking
    )

    add_test(
        NAME trajectory_io
        COMMAND trajectory_io_tests
    )

    add_executable(trajectory_analysis_tests
        tests/test_trajectory_analysis.cpp
    )

    target_link_libraries(trajectory_analysis_tests
        PRIVATE
            tracking
    )

    add_test(
        NAME trajectory_analysis
        COMMAND trajectory_analysis_tests
    )
endif()

option(TRACKING_BUILD_BENCHMARKS "Build prediction benchmarks" OFF)
if(TRACKING_BUILD_BENCHMARKS)
    if(TRACKING_ENABLE_SANITIZERS)
        message(FATAL_ERROR "Prediction benchmarks require sanitizers to be disabled.")
    endif()
    add_executable(prediction_benchmark benchmarks/prediction_benchmark.cpp)
    target_link_libraries(prediction_benchmark PRIVATE tracking)
endif()

get_property(tracking_project_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
foreach(tracking_target IN LISTS tracking_project_targets)
    get_target_property(tracking_target_type ${tracking_target} TYPE)
    if(tracking_target_type MATCHES "^(EXECUTABLE|STATIC_LIBRARY|SHARED_LIBRARY|OBJECT_LIBRARY)$")
        target_link_libraries(${tracking_target} PRIVATE project_warnings project_sanitizers)
        if(TRACKING_ENABLE_SANITIZERS AND MSVC AND tracking_target_type STREQUAL "EXECUTABLE")
            foreach(tracking_asan_runtime IN LISTS tracking_asan_runtimes)
                add_custom_command(TARGET ${tracking_target} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${tracking_asan_runtime}" "$<TARGET_FILE_DIR:${tracking_target}>"
                    VERBATIM
                )
            endforeach()
        endif()
    endif()
endforeach()
