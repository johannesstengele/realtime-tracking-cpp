option(TRACKING_WARNINGS_AS_ERRORS "Treat project warnings as errors" OFF)

add_library(project_warnings INTERFACE)

if(MSVC)
    target_compile_options(project_warnings INTERFACE /W4 /permissive-)
    if(TRACKING_WARNINGS_AS_ERRORS)
        target_compile_options(project_warnings INTERFACE /WX)
    endif()
else()
    target_compile_options(project_warnings INTERFACE
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion
    )
    if(TRACKING_WARNINGS_AS_ERRORS)
        target_compile_options(project_warnings INTERFACE -Werror)
    endif()
endif()

option(TRACKING_ENABLE_SANITIZERS "Enable runtime sanitizer instrumentation" OFF)
add_library(project_sanitizers INTERFACE)

if(TRACKING_ENABLE_SANITIZERS)
    if(MSVC)
        target_compile_options(project_sanitizers INTERFACE /fsanitize=address)
        target_link_options(project_sanitizers INTERFACE /INCREMENTAL:NO /INFERASANLIBS)
        get_filename_component(tracking_compiler_directory "${CMAKE_CXX_COMPILER}" DIRECTORY)
        file(GLOB tracking_asan_runtimes "${tracking_compiler_directory}/clang_rt.asan*_dynamic-*.dll")
        if(NOT tracking_asan_runtimes)
            message(FATAL_ERROR "The selected MSVC installation does not provide AddressSanitizer.")
        endif()
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(project_sanitizers INTERFACE
            -fsanitize=address,undefined -fno-omit-frame-pointer
        )
        target_link_options(project_sanitizers INTERFACE
            -fsanitize=address,undefined -fno-omit-frame-pointer
        )
    else()
        message(FATAL_ERROR "Sanitizers are not configured for this compiler.")
    endif()
endif()
