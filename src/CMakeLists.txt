
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cuda-samples/Common)

hip_add_executable(c15
    starter_other/ConstMemory.cu
)

hip_add_executable(mt4g
    capture.cu
)

if(COMPUTE_CAP_COMPILE_OPTIONS)
    target_compile_options(c15 ${COMPUTE_CAP_COMPILE_OPTIONS})
    target_compile_options(mt4g ${COMPUTE_CAP_COMPILE_OPTIONS})
endif()

target_link_libraries(c15 PRIVATE hip::device)
target_link_libraries(mt4g PRIVATE hip::device)

if(HIP_PLATFORM STREQUAL "nvidia")
    message(STATUS "Linking for NVIDIA environment. Provide any NVIDIA-specific libs as needed.")
endif()

if(UNIX)
    target_link_libraries(c15 PRIVATE "stdc++fs")
    target_link_libraries(mt4g PRIVATE "stdc++fs")
endif()

install(TARGETS mt4g c15 DESTINATION bin)
