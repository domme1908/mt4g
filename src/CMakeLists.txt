# src/CMakeLists.txt

# We assume we are using "hipcc" as the compiler at the top-level.
# So we just build executables as normal C++ targets.

# Example: Add extra include dirs if needed
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cuda-samples/Common)

# The user wants 2 executables: c15 and mt4g
add_executable(c15
    starter_other/ConstMemory.cu
    # ... If there are other source files for c15, list them here ...
)

add_executable(mt4g
    capture.cu
    # ... If there are additional .cu/.cpp files for mt4g, list them here ...
)

# Add architecture flags from the top-level CMakeLists
# By default, these go into compiler options for *all* sources in each target.
if(COMPUTE_CAP_COMPILE_OPTIONS)
    target_compile_options(c15 PRIVATE ${COMPUTE_CAP_COMPILE_OPTIONS})
    target_compile_options(mt4g PRIVATE ${COMPUTE_CAP_COMPILE_OPTIONS})
endif()

# If you need special libraries, link them. For example:
#   target_link_libraries(<target> <libs...>)
# For basic HIP code, you often don't need anything special unless you use
# device-level or driver-level calls that require -lcuda, etc.

# If you still want to link with "stdc++fs" or other:
if(UNIX)
    target_link_libraries(c15 PRIVATE stdc++fs)
    target_link_libraries(mt4g PRIVATE stdc++fs)
endif()

# If you want NVML for nvidia:
if(HIP_PLATFORM STREQUAL "nvidia")
    # e.g., target_link_libraries(mt4g PRIVATE "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so")
endif()

# If you want to install them somewhere:
install(TARGETS c15 mt4g DESTINATION bin)
