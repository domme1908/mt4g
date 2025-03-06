
#ifndef CUDATEST_MAIN
#define CUDATEST_MAIN

#include <cstdio>

#include "eval.hip.h"
#include "GPU_resources.hip.h"

__global__ void main_size_test(unsigned int *my_array, unsigned int *duration, unsigned int *index, bool *isDisturbed);

bool launchMainKernelBenchmark(int N, int stride, double *avgOut, unsigned int *potMissesOut, unsigned int **time, int *error);

#define FreeMeasureMain() \
    free(avg);            \
    free(misses);         \
    free(time);

CacheResults measure_Main(int l2SizeInBytes, int stride)
{
    double *avg = (double *)malloc(sizeof(double));
    unsigned int *misses = (unsigned int *)malloc(sizeof(unsigned int));
    unsigned int **time = (unsigned int **)malloc(sizeof(unsigned int *));
    if (avg == nullptr || misses == nullptr || time == nullptr)
    {
        FreeMeasureMain()
            printErrorCodeInformation(1);
        exit(1);
    }

    int error = 0;
    bool dist = true;
    int count = 5;

    while (dist && count > 0)
    {
        dist = launchMainKernelBenchmark(l2SizeInBytes, stride, avg, misses, time, &error);
        --count;
    }

    free(time[0]);
    FreeMeasureMain()

        if (error != 0)
    {
        printErrorCodeInformation(error);
        exit(error);
    }

    return CacheResults{};
}

bool launchMainKernelBenchmark(int N, int stride, double *avgOut, unsigned int *potMissesOut, unsigned int **time, int *error)
{
    hipError_t error_id;

    unsigned int *h_a = nullptr, *h_index = nullptr, *h_timeinfo = nullptr,
                 *d_a = nullptr, *d_index = nullptr, *duration = nullptr;
    bool *disturb = nullptr, *d_disturb = nullptr;

    do
    {
        // Allocate Memory on Host
        h_a = (unsigned int *)malloc(sizeof(unsigned int) * (N));
        if (h_a == nullptr)
        {
            printf("[MAINMEMTEST.CUH]: malloc h_a Error\n");
            *error = 1;
            break;
        }

        h_index = (unsigned int *)malloc(sizeof(unsigned int) * MEASURE_SIZE);
        if (h_index == nullptr)
        {
            printf("[MAINMEMTEST.CUH]: malloc h_index Error\n");
            *error = 1;
            break;
        }

        h_timeinfo = (unsigned int *)malloc(sizeof(unsigned int) * MEASURE_SIZE);
        if (h_timeinfo == nullptr)
        {
            printf("[MAINMEMTEST.CUH]: malloc h_timeinfo Error\n");
            *error = 1;
            break;
        }

        disturb = (bool *)malloc(sizeof(bool));
        if (disturb == nullptr)
        {
            printf("[MAINMEMTEST.CUH]: malloc disturb Error\n");
            *error = 1;
            break;
        }

        // Allocate Memory on GPU
        error_id = hipMalloc((void **)&d_a, sizeof(unsigned int) * (N));
        if (error_id != hipSuccess)
        {
            printf("[MAINMEMTEST.CUH]: hipMalloc d_a Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = hipMalloc((void **)&duration, sizeof(unsigned int) * MEASURE_SIZE);
        if (error_id != hipSuccess)
        {
            printf("[MAINMEMTEST.CUH]: hipMalloc duration Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = hipMalloc((void **)&d_index, sizeof(unsigned int) * MEASURE_SIZE);
        if (error_id != hipSuccess)
        {
            printf("[MAINMEMTEST.CUH]: hipMalloc d_index Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = hipMalloc((void **)&d_disturb, sizeof(bool));
        if (error_id != hipSuccess)
        {
            printf("[MAINMEMTEST.CUH]: hipMalloc d_disturb Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }

        // Initialize p-chase array
        for (int i = 0; i < N; i++)
        {
            // original:
            h_a[i] = (i + stride) % N;
        }

        // Copy array from Host to GPU
        error_id = hipMemcpy(d_a, h_a, sizeof(unsigned int) * N, hipMemcpyHostToDevice);
        if (error_id != hipSuccess)
        {
            printf("[MAINMEMTEST.CUH]: hipMemcpy h_a Error: %s\n", hipGetErrorString(error_id));
            *error = 3;
            break;
        }

        (void)hipDeviceSynchronize();

        // Launch Kernel function
        dim3 Db = dim3(1);
        dim3 Dg = dim3(1, 1, 1);
        main_size_test<<<Dg, Db>>>(d_a, duration, d_index, d_disturb);

        (void)hipDeviceSynchronize();

        error_id = hipGetLastError();
        if (error_id != hipSuccess)
        {
            printf("[MAINMEMTEST.CUH]: Kernel launch/execution with clock Error:%s\n", hipGetErrorString(error_id));
            *error = 5;
            break;
        }
        (void)hipDeviceSynchronize();

        // Copy results from GPU to Host
        error_id = hipMemcpy((void *)h_timeinfo, (void *)duration, sizeof(unsigned int) * MEASURE_SIZE, hipMemcpyDeviceToHost);
        if (error_id != hipSuccess)
        {
            printf("[MAINMEMTEST.CUH]: hipMemcpy duration Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = hipMemcpy((void *)h_index, (void *)d_index, sizeof(unsigned int) * MEASURE_SIZE, hipMemcpyDeviceToHost);
        if (error_id != hipSuccess)
        {
            printf("[MAINMEMTEST.CUH]: hipMemcpy d_index Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = hipMemcpy((void *)disturb, (void *)d_disturb, sizeof(bool), hipMemcpyDeviceToHost);
        if (error_id != hipSuccess)
        {
            printf("[MAINMEMTEST.CUH]: hipMemcpy d_disturb Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }
        (void)hipDeviceSynchronize();

        createOutputFile(N, MEASURE_SIZE, h_index, h_timeinfo, avgOut, potMissesOut, "Main_");
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

    // Free Memory on Host
    bool ret = false;
    if (disturb != nullptr)
    {
        ret = *disturb;
        free(disturb);
    }

    if (h_a != nullptr)
    {
        free(h_a);
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

    (void)hipDeviceReset();
    return ret;
}

__global__ void main_size_test(unsigned int *my_array, unsigned int *duration, unsigned int *index, bool *isDisturbed)
{

    unsigned int start_time, end_time;

    bool dist = false;
    unsigned int j = 0;

    for (int k = 0; k < MEASURE_SIZE; k++)
    {
        s_index[k] = 0;
        s_tvalue[k] = 0;
    }

    // Warming up, filling TLP and PT
    for (int k = 0; k < 32; k++)
    {
        j = my_array[j];
    }

#ifdef IS_AMD
    uint64_t smem_ptr64 = 0; // Initialize LDS pointer
    // Initialize the LDS pointer to 0
    asm volatile("s_mov_b64 %0, 0\n\t" : "=s"(smem_ptr64));
#else
    // No first round required
    asm volatile(
        " .reg .u64 smem_ptr64;\n\t"
        " cvta.to.shared.u64 smem_ptr64, %0;\n\t"
        : : "r"(s_index));
#endif
    for (int k = 0; k < MEASURE_SIZE; k++)
    {
        unsigned int *ptr = my_array + j;
#ifdef IS_AMD
        uint64_t start_time, end_time;
        uint32_t j;
        uint32_t v_smem_ptr;

        asm volatile("s_memtime %0" : "=r"(start_time));

        asm volatile("flat_load_dword %0, %1\n\t" : "=v"(j) : "v"(ptr));

        asm volatile("v_mov_b32 %0, 0\n\t" : "=v"(v_smem_ptr));

        asm volatile("ds_write_b32 %0, %1\n\t" : : "v"(v_smem_ptr), "v"(j));

        asm volatile("s_memtime %0" : "=r"(end_time));

        asm volatile("v_add_u32 %0, %0, 4\n\t" : "+v"(v_smem_ptr));
#else
        asm volatile("mov.u32 %0, %%clock;\n\t"
                     "ld.global.cg.u32 %1, [%3];\n\t"
                     "st.shared.u32 [smem_ptr64], %1;"
                     "mov.u32 %2, %%clock;\n\t"
                     "add.u64 smem_ptr64, smem_ptr64, 4;" : "=r"(start_time), "=r"(j), "=r"(end_time) : "r"(ptr) : "memory");
#endif
        s_tvalue[k] = end_time - start_time;
    }

    for (int k = 0; k < MEASURE_SIZE; k++)
    {
        if (s_tvalue[k] > 1200)
        {
            dist = true;
        }
        index[k] = s_index[k];
        duration[k] = s_tvalue[k];
    }
    *isDisturbed = dist;
}

#endif // CUDATEST_MAIN
