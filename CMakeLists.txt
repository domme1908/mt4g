cmake_minimum_required(VERSION 3.22)
project(mt4g LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#
# 1. Force the compiler to hipcc, so we skip find_package(HIP).
#    This is the critical step if your HIP install doesn't provide HIPConfig.cmake.
#
set(CMAKE_CXX_COMPILER "/usr/bin/hipcc")

#
# 2. Expose a HIP_PLATFORM variable (amd or nvidia).
#    The user can do: cmake -B build -DHIP_PLATFORM=amd  (or nvidia).
#
if(NOT DEFINED HIP_PLATFORM)
    # Default to 'nvidia' if not specified
    set(HIP_PLATFORM "nvidia")
endif()

message(STATUS "HIP_PLATFORM = ${HIP_PLATFORM}")

#
# 3. Similarly, let the user specify GPU architecture:
#    - For NVIDIA:  -DNVIDIA_UARCH=TURING (or AMPERE, etc.)
#    - For AMD:     -DAMD_UARCH=VEGA20 (etc.)
#
if(DEFINED NVIDIA_UARCH)
    string(TOUPPER "${NVIDIA_UARCH}" NVIDIA_UARCH)
    message(STATUS "NVIDIA_UARCH = ${NVIDIA_UARCH}")
endif()

if(DEFINED AMD_UARCH)
    string(TOUPPER "${AMD_UARCH}" AMD_UARCH)
    message(STATUS "AMD_UARCH = ${AMD_UARCH}")
endif()

#
# 4. Construct compile flags for either AMD or NVIDIA, just like your old logic.
#
set(COMPUTE_CAP_COMPILE_OPTIONS "")

if(HIP_PLATFORM STREQUAL "nvidia")
    # Mirror your original architecture checks
    if(NVIDIA_UARCH STREQUAL "KEPLER")
        set(COMPUTE_CAP_COMPILE_OPTIONS "--gencode=arch=compute_35,code=sm_35")
    elseif(NVIDIA_UARCH STREQUAL "MAXWELL")
        set(COMPUTE_CAP_COMPILE_OPTIONS "--gencode=arch=compute_50,code=sm_50")
    elseif(NVIDIA_UARCH STREQUAL "PASCAL")
        set(COMPUTE_CAP_COMPILE_OPTIONS "--gencode=arch=compute_60,code=sm_60")
    elseif(NVIDIA_UARCH STREQUAL "VOLTA")
        set(COMPUTE_CAP_COMPILE_OPTIONS "--gencode=arch=compute_70,code=sm_70")
    elseif(NVIDIA_UARCH STREQUAL "TURING")
        set(COMPUTE_CAP_COMPILE_OPTIONS "--gencode=arch=compute_75,code=sm_75")
    elseif(NVIDIA_UARCH STREQUAL "AMPERE")
        set(COMPUTE_CAP_COMPILE_OPTIONS "--gencode=arch=compute_86,code=sm_86")
    elseif(NVIDIA_UARCH STREQUAL "HOPPER")
        set(COMPUTE_CAP_COMPILE_OPTIONS "--gencode=arch=compute_90,code=sm_90")
    else()
        message(STATUS "No (or unknown) NVIDIA_UARCH specified. Using default hipcc/nvcc arch.")
    endif()

elseif(HIP_PLATFORM STREQUAL "amd")
    # Example AMD arches from last ~3-4 years
    if(AMD_UARCH STREQUAL "VEGA20")
        set(COMPUTE_CAP_COMPILE_OPTIONS "--amdgpu-target=gfx906")
    elseif(AMD_UARCH STREQUAL "ARCTURUS")
        set(COMPUTE_CAP_COMPILE_OPTIONS "--amdgpu-target=gfx908")
    elseif(AMD_UARCH STREQUAL "ALDEBARAN")
        set(COMPUTE_CAP_COMPILE_OPTIONS "--amdgpu-target=gfx90a")
    elseif(AMD_UARCH STREQUAL "NAVI21")
        set(COMPUTE_CAP_COMPILE_OPTIONS "--amdgpu-target=gfx1030")
    elseif(AMD_UARCH STREQUAL "RDNA3")
        set(COMPUTE_CAP_COMPILE_OPTIONS "--amdgpu-target=gfx1100")
    else()
        message(STATUS "No (or unknown) AMD_UARCH specified. Using default hipcc/clang arch.")
    endif()
else()
    message(WARNING "HIP_PLATFORM neither 'nvidia' nor 'amd'; no arch flags set.")
endif()

# We'll pass these flags to the compiler in the subdir. Mark them CACHE or just keep them normal.
set(COMPUTE_CAP_COMPILE_OPTIONS "${COMPUTE_CAP_COMPILE_OPTIONS}" CACHE STRING "HIP architecture flags" FORCE)

#
# 5. Add subdirectory to build the actual sources.
#
add_subdirectory(src)
