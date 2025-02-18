#ifndef CUDATEST_L1
#define CUDATEST_L1

#include <cstdio>
#include <chrono>
#include "binarySearch.h"
#include <hip/hip_runtime.h>
#include "eval.h"
#include "GPU_resources.hip.cu"
# include "utils.h"

__global__ void l1_size(unsigned int *my_array, int array_length, unsigned int *duration, unsigned int *index, bool *isDisturbed);

bool launchL1KernelBenchmark(int N, int stride, double *avgOut, unsigned int *potMissesOut, unsigned int **time, int *error);
char unitsByteLocal[4][4] = {"B", "KiB", "MiB", "GiB"};

const char* formatSize(double* val, size_t original) {
    int unitIndex = 0;

    if (original > 1024 * 1024 * 1024) {
        original = original >> 10;
        ++unitIndex;
    }

    double result = (double) original;

    if (result > 1000.) {
        result = result / 1024.;
        ++unitIndex;
    }

    if (result > 1000.) {
        result = result / 1024.;
        ++unitIndex;
    }

    const char* unit = unitsByteLocal[unitIndex];
    *val = result;
    return unit;
}
CacheSizeResult measure_L1()
{
    auto startTime = std::chrono::high_resolution_clock::now();
    // 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
    int absoluteLowerBoundary = 1024;
    int absoluteUpperBoundary = 1024 << 10; // 1024 * 1024
    int widenBounds = 8;

    // Start with 1K integers until 1000K integers
    int bounds[2] = {absoluteLowerBoundary, absoluteUpperBoundary};
    getBoundaries(launchL1KernelBenchmark, bounds, 5);
#ifdef IsDebug
    fprintf(out, "Got Boundaries: %d...%d\n", bounds[0], bounds[1]);
#endif // IsDebug
    printf("Got Boundaries: %d...%d\n", bounds[0], bounds[1]);

    int cp = -1;
    int begin = bounds[0] - widenBounds;
    int end = bounds[1] + widenBounds;
    int stride = 8;
    int arrayIncrease = 8;

    while (cp == -1 && begin >= absoluteLowerBoundary / sizeof(int) - widenBounds && end <= absoluteUpperBoundary / sizeof(int) + widenBounds)
    {
        cp = wrapBenchmarkLaunch(launchL1KernelBenchmark, begin, end, stride, arrayIncrease, "L1");

        if (cp == -1)
        {
            begin = begin - (end - begin);
            end = end + (end - begin);
#ifdef IsDebug
            fprintf(out, "\nGot Boundaries: %d...%d\n", begin, end);
#endif // IsDebug
            printf("\nGot Boundaries: %d...%d\n", begin, end);
        }
    }

    CacheSizeResult result;
    int cacheSizeInInt = (begin + cp * arrayIncrease);
    result.CacheSize = (cacheSizeInInt << 2);
    result.realCP = cp > 0;
    result.maxSizeBenchmarked = end << 2;
    auto endTime = std::chrono::high_resolution_clock::now();
    printf("measure_L1 time: %f ms\n", std::chrono::duration<double>(endTime - startTime).count() * 1000.0);
    double size;
    size_t original = result.CacheSize;
    const char* unit = formatSize(&size, original);
    printf("Size %f%s\n", size, unit);
    printf("Stride was %d\n", stride);
    printf("ArrayIncrease was %d\n", arrayIncrease);
    return result;
}

bool launchL1KernelBenchmark(int N, int stride, double *avgOut, unsigned int *potMissesOut, unsigned int **time, int *error)
{
    // hipDeviceReset();
    hipError_t error_id;
    unsigned int *h_a = nullptr, *h_index = nullptr, *h_timeinfo = nullptr,
                 *d_a = nullptr, *duration = nullptr, *d_index = nullptr, *lines = nullptr;
    bool *disturb = nullptr, *d_disturb = nullptr;

    do
    {
        // Allocate Memory on Host
        h_a = (unsigned int *)malloc(sizeof(unsigned int) * (N));
        if (h_a == nullptr)
        {
            printf("[L1.CUH]: malloc h_a Error\n");
            *error = 1;
            break;
        }

        h_index = (unsigned int *)malloc(sizeof(unsigned int) * MEASURE_SIZE);
        if (h_index == nullptr)
        {
            printf("[L1.CUH]: malloc h_index Error\n");
            *error = 1;
            break;
        }

        h_timeinfo = (unsigned int *)malloc(sizeof(unsigned int) * MEASURE_SIZE);
        if (h_timeinfo == nullptr)
        {
            printf("[L1.CUH]: malloc h_timeinfo Error\n");
            *error = 1;
            break;
        }

        disturb = (bool *)malloc(sizeof(bool));
        if (disturb == nullptr)
        {
            printf("[L1.CUH]: malloc disturb Error\n");
            *error = 1;
            break;
        }

        // Allocate Memory on GPU
        error_id = hipMalloc((void **)&d_a, sizeof(unsigned int) * (N));
        if (error_id != hipSuccess)
        {
            printf("[L1.CUH]: hipMalloc d_a Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = hipMalloc((void **)&duration, sizeof(unsigned int) * MEASURE_SIZE);
        if (error_id != hipSuccess)
        {
            printf("[L1.CUH]: hipMalloc duration Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = hipMalloc((void **)&d_index, sizeof(unsigned int) * MEASURE_SIZE);
        if (error_id != hipSuccess)
        {
            printf("[L1.CUH]: hipMalloc d_index Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = hipMalloc((void **)&d_disturb, sizeof(bool));
        if (error_id != hipSuccess)
        {
            printf("[L1.CUH]: hipMalloc disturb Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }

        int line_count = N / stride;

        lines = (unsigned int *)malloc(sizeof(unsigned int) * line_count);
        if (!lines)
        {
            printf("Error: malloc for lines failed.\n");
            free(h_a);
            return 1;
        }

        for (int i = 0; i < line_count; i++)
        {
            lines[i] = i;
        }

        fisher_yates_shuffle(lines, line_count);
        for (int i = 0; i < line_count - 1; i++)
        {
            int current_line = lines[i];
            int next_line = lines[i + 1];
            h_a[current_line * stride] = next_line * stride;
        }
        h_a[lines[line_count - 1] * stride] = lines[0] * stride;

        // Copy array from Host to GPU
        error_id = hipMemcpy(d_a, h_a, sizeof(unsigned int) * N, hipMemcpyHostToDevice);
        if (error_id != hipSuccess)
        {
            printf("[L1.CUH]: hipMemcpy d_a Error: %s\n", hipGetErrorString(error_id));
            *error = 3;
            break;
        }
        hipDeviceSynchronize();

        // Launch Kernel function
        // Single thread i think
        dim3 Db = dim3(1);
        dim3 Dg = dim3(1, 1, 1);
        hipLaunchKernelGGL(l1_size, Dg, Db, 0, 0, d_a, N, duration, d_index, d_disturb);

        hipDeviceSynchronize();

        error_id = hipGetLastError();
        if (error_id != hipSuccess)
        {
            printf("[L1.CUH]: Kernel launch/execution Error: %s\n", hipGetErrorString(error_id));
            *error = 5;
            break;
        }
        hipDeviceSynchronize();

        // Copy results from GPU to Host
        error_id = hipMemcpy((void *)h_timeinfo, (void *)duration, sizeof(unsigned int) * MEASURE_SIZE, hipMemcpyDeviceToHost);
        if (error_id != hipSuccess)
        {
            printf("[L1.CUH]: hipMemcpy duration Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = hipMemcpy((void *)h_index, (void *)d_index, sizeof(unsigned int) * MEASURE_SIZE, hipMemcpyDeviceToHost);
        if (error_id != hipSuccess)
        {
            printf("[L1.CUH]: hipMemcpy d_index Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = hipMemcpy((void *)disturb, (void *)d_disturb, sizeof(bool), hipMemcpyDeviceToHost);
        if (error_id != hipSuccess)
        {
            printf("[L1.CUH]: hipMemcpy disturb Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }

        hipDeviceSynchronize();

        if (!*disturb)
            createOutputFile(N, MEASURE_SIZE, h_index, h_timeinfo, avgOut, potMissesOut, "L1_");

    } while (false);

    // Free Memory on GPU
    if (d_a != nullptr)
    {
        hipFree(d_a);
    }

    if (d_index != nullptr)
    {
        hipFree(d_index);
    }

    if (duration != nullptr)
    {
        hipFree(duration);
    }

    if (d_disturb != nullptr)
    {
        hipFree(d_disturb);
    }

    bool ret = false;
    if (disturb != nullptr)
    {
        ret = *disturb;
        free(disturb);
    }

    // Free Memory on Host
    if (h_a != nullptr)
    {
        free(h_a);
    }
    if (lines != nullptr)
    {
        free(lines);
    }

    if (h_index != nullptr)
    {
        free(h_index);
    }

    if (h_timeinfo != nullptr)
    {
        if (time != nullptr)
        {
            time[0] = h_timeinfo;
        }
        else
        {
            free(h_timeinfo);
        }
    }

    hipDeviceReset();
    return ret;
}

__global__ void l1_size(unsigned int *my_array, int array_length, unsigned int *duration, unsigned int *index, bool *isDisturbed)
{

    unsigned int start_time, end_time;
    bool dist = false;
    unsigned int j = 0;

    for (int k = 0; k < MEASURE_SIZE; k++)
    {
        s_index[k] = 0;
        s_tvalue[k] = 0;
    }

    // First round
    unsigned int *ptr;
    for (int k = 0; k < array_length; k++)
    {
        ptr = my_array + j;
#ifdef IS_AMD
        asm volatile("ld.global.u32 %0, [%1];" : "=r"(j) : "l"(ptr) : "memory");
#else
        asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(j) : "l"(ptr) : "memory");
#endif
        // j = my_array[j];
    }

    // Second round
#ifdef IS_AMD
    asm volatile(
        // Declare register
        " // no-op for AMD pointer conversion\n\t"
        ::"l"(s_index));
#else
    asm volatile(
        // Declare register
        " .reg .u64 smem_ptr64;\n\t"
        // Convert a c pointer into a shared memory address - I think
        " cvta.to.shared.u64 smem_ptr64, %0;\n\t"
        ::"l"(s_index));
#endif
    for (int k = 0; k < MEASURE_SIZE; k++)
    {
        ptr = my_array + j;
#ifdef IS_AMD
        asm volatile(
            // Save GPU Clock into start_time var
            "s_memtime s2:s3;\n\t"
            // Load ptr into register
            "ld.global.u32 %1, [%3];\n\t"
            // Write data to shared memory
            "st.shared.u32 [smem_ptr64], %1;"
            // Save GPU Clock into end_time var
            "s_memtime s4:s5;\n\t"
            // Increment shared memory pointer by 4 bytes
            "add.u64 smem_ptr64, smem_ptr64, 4;" : "=r"(start_time), "=r"(j), "=r"(end_time) : "l"(ptr) : "memory");
#else
        asm volatile(
            // Save GPU Clock into start_time var
            "mov.u32 %0, %%clock;\n\t"
            // Load ptr into register
            "ld.global.ca.u32 %1, [%3];\n\t"
            // Write data to shared memory
            "st.shared.u32 [smem_ptr64], %1;"
            // Save GPU Clock into end_time var
            "mov.u32 %2, %%clock;\n\t"
            // Increment shared memory pointer by 4 bytes
            "add.u64 smem_ptr64, smem_ptr64, 4;" : "=r"(start_time), "=r"(j), "=r"(end_time) : "l"(ptr) : "memory");
#endif
        s_tvalue[k] = end_time - start_time;
    }

    for (int k = 0; k < MEASURE_SIZE; k++)
    {
        if (s_tvalue[k] > 2000)
        {
            dist = true;
        }
        index[k] = s_index[k];
        duration[k] = s_tvalue[k];
    }
    *isDisturbed = dist;
}

#endif // CUDATEST_L1
