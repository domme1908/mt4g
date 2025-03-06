//
// Created by nick- on 6/29/2022.
//

#ifndef CUDATEST_L1_L2_DIFF_CUH
#define CUDATEST_L1_L2_DIFF_CUH

#include <cstdio>

#define diffSize 100

__global__ void l1_differ(unsigned int *my_array, unsigned int *durationL1, unsigned int *indexL1)
{
    uint64_t start_time, end_time;

    __shared__ long long s_tvalue_l1[diffSize];
    __shared__ unsigned int s_index_l1[diffSize];

    for (int k = 0; k < diffSize; k++)
    {
        s_index_l1[k] = 0;
        s_tvalue_l1[k] = 0;
    }

    unsigned int j = 0;
    unsigned int *ptr;

    // First round
    for (int k = 0; k < diffSize; k++)
    {
        ptr = my_array + j;
#ifdef IS_AMD
        asm volatile(
            "global_load_dword %0, %1, off"
            : "=v"(j)
            : "v"(ptr));
#else
        asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(j) : "r"(ptr) : "memory");
#endif
    }
#ifdef IS_AMD
    uint64_t smem_ptr64 = 0; // Initialize LDS pointer
    // Initialize the LDS pointer to 0
    asm volatile("s_mov_b64 %0, 0\n\t" : "=s"(smem_ptr64));
#else

    // Second round
    asm volatile(" .reg .u64 smem_ptr64;\n\t"
                 " cvta.to.shared.u64 smem_ptr64, %0;\n\t" ::"r"(s_index_l1));
#endif
    for (int k = 0; k < diffSize; k++)
    {
        ptr = my_array + j;
#ifdef IS_AMD
        uint32_t start_time_lo, start_time_hi;
        uint32_t j;
        uint32_t v_smem_ptr;

        // Read GPU clock into two 32-bit registers
        asm volatile("s_memtime %0" : "=r"(start_time));

        // Load from global memory (flat memory addressing)
        asm volatile("flat_load_dword %0, %1\n\t" : "=v"(j) : "v"(ptr));

        // Initialize VGPR LDS pointer (INSTEAD of SGPR)
        asm volatile("v_mov_b32 %0, 0\n\t" : "=v"(v_smem_ptr));

        // Store to LDS (shared memory)
        asm volatile("ds_write_b32 %0, %1\n\t" : : "v"(v_smem_ptr), "v"(j));

        // Read GPU clock again
        asm volatile("s_memtime %0" : "=r"(end_time));

        // Increment shared memory pointer
        asm volatile("v_add_u32 %0, %0, 4\n\t" : "+v"(v_smem_ptr));
#else
        asm volatile(
            "mov.u32 %0, %%clock;\n\t"
            "ld.global.ca.u32 %1, [%3];\n\t"
            "st.shared.u32 [smem_ptr64], %1;"
            "mov.u32 %2, %%clock;\n\t"
            "add.u64 smem_ptr64, smem_ptr64, 4;" : "=r"(start_time), "=r"(j), "=r"(end_time) : "r"(ptr) : "memory");
#endif
        s_tvalue_l1[k] = end_time - start_time;
    }

    for (int k = 0; k < diffSize; k++)
    {
        indexL1[k] = s_index_l1[k];
        durationL1[k] = s_tvalue_l1[k];
    }
}

__global__ void l2_differ(unsigned int *my_array, unsigned int *durationL2, unsigned int *indexL2)
{
    uint64_t start_time, end_time;

    __shared__ long long s_tvalue_l2[diffSize];
    __shared__ unsigned int s_index_l2[diffSize];

    for (int k = 0; k < diffSize; k++)
    {
        s_index_l2[k] = 0;
        s_tvalue_l2[k] = 0;
    }

    unsigned int j = 0;
    unsigned int *ptr;

    // First round
    for (int k = 0; k < diffSize; k++)
    {
        ptr = my_array + j;
#ifdef IS_AMD
        asm volatile(
            "global_load_dword %0, %1, off\n\t"
            : "=v"(j)
            : "v"(ptr));
#else
        asm volatile("ld.global.cg.u32 %0, [%1];\n\t" : "=r"(j) : "r"(ptr) : "memory");
#endif
    }

    // Second round
    for (int k = 0; k < diffSize; k++)
    {
        ptr = my_array + j;
        start_time = clock();
#ifdef IS_AMD
        asm volatile("s_memtime %0" : "=r"(start_time));
        asm volatile("flat_load_dword %0, %1\n\t" : "=v"(j) : "v"(ptr));
#else
        asm volatile("mov.u32 %0, %%clock;\n\t"
                     "ld.global.cg.u32 %1, [%2];\n\t" : "=r"(start_time), "=r"(j) : "r"(ptr) : "memory");
#endif
        s_index_l2[k] = j;
#ifdef IS_AMD
        asm volatile("s_memtime %0" : "=r"(end_time));
#else
        asm volatile("mov.u32 %0, %%clock;\n\t" : "=r"(end_time));
#endif
        s_tvalue_l2[k] = end_time - start_time;
    }

    for (int k = 0; k < diffSize; k++)
    {
        indexL2[k] = s_index_l2[k];
        durationL2[k] = s_tvalue_l2[k];
    }
}

bool measureL1_L2_difference(double tol)
{
    int error = 0;
    unsigned int *h_a = nullptr, *h_indexL1 = nullptr, *h_timeinfoL1 = nullptr, *h_indexL2 = nullptr, *h_timeinfoL2 = nullptr,
                 *d_a = nullptr, *durationL1 = nullptr, *durationL2 = nullptr, *d_indexL1 = nullptr, *d_indexL2 = nullptr;
    double absDistance = 0.;

    do
    {
        // Allocate Memory on Host
        h_a = (unsigned int *)malloc(sizeof(unsigned int) * diffSize);
        if (h_a == nullptr)
        {
            printf("[L1_L2_DIFF.CUH]: malloc h_a Error\n");
            error = 1;
            break;
        }

        h_indexL1 = (unsigned int *)malloc(sizeof(unsigned int) * diffSize);
        if (h_indexL1 == nullptr)
        {
            printf("[L1_L2_DIFF.CUH]: malloc h_indexL1 Error\n");
            error = 1;
            break;
        }

        h_timeinfoL1 = (unsigned int *)malloc(sizeof(unsigned int) * diffSize);
        if (h_timeinfoL1 == nullptr)
        {
            printf("[L1_L2_DIFF.CUH]: malloc h_timeinfoL1 Error\n");
            error = 1;
            break;
        }

        h_indexL2 = (unsigned int *)malloc(sizeof(unsigned int) * diffSize);
        if (h_indexL2 == nullptr)
        {
            printf("[L1_L2_DIFF.CUH]: malloc h_indexL2 Error\n");
            error = 1;
            break;
        }

        h_timeinfoL2 = (unsigned int *)malloc(sizeof(unsigned int) * diffSize);
        if (h_timeinfoL2 == nullptr)
        {
            printf("[L1_L2_DIFF.CUH]: malloc h_timeinfoL2 Error\n");
            error = 1;
            break;
        }

        // Allocate Memory on GPU
        hipError_t error_id = hipMalloc((void **)&d_a, sizeof(unsigned int) * diffSize);
        if (error_id != hipSuccess)
        {
            printf("[L1_L2_DIFF.CUH]: hipMalloc d_a Error: %s\n", hipGetErrorString(error_id));
            error = 2;
            break;
        }

        error_id = hipMalloc((void **)&durationL1, sizeof(unsigned int) * diffSize);
        if (error_id != hipSuccess)
        {
            printf("[L1_L2_DIFF.CUH]: hipMalloc durationL1 Error: %s\n", hipGetErrorString(error_id));
            error = 2;
            break;
        }

        error_id = hipMalloc((void **)&durationL2, sizeof(unsigned int) * diffSize);
        if (error_id != hipSuccess)
        {
            printf("[L1_L2_DIFF.CUH]: hipMalloc durationL2 Error: %s\n", hipGetErrorString(error_id));
            error = 2;
            break;
        }

        error_id = hipMalloc((void **)&d_indexL1, sizeof(unsigned int) * diffSize);
        if (error_id != hipSuccess)
        {
            printf("[L1_L2_DIFF.CUH]: hipMalloc d_indexL1 Error: %s\n", hipGetErrorString(error_id));
            error = 2;
            break;
        }

        error_id = hipMalloc((void **)&d_indexL2, sizeof(unsigned int) * diffSize);
        if (error_id != hipSuccess)
        {
            printf("[L1_L2_DIFF.CUH]: hipMalloc d_indexL2 Error: %s\n", hipGetErrorString(error_id));
            error = 2;
            break;
        }

        // Initialize p-chase array
        for (int i = 0; i < diffSize; i++)
        {
            h_a[i] = (i + 1) % diffSize;
        }

        // Copy array from Host to GPU
        error_id = hipMemcpy(d_a, h_a, sizeof(unsigned int) * diffSize, hipMemcpyHostToDevice);
        if (error_id != hipSuccess)
        {
            printf("[L1_L2_DIFF.CUH]: hipMemcpy d_a Error: %s\n", hipGetErrorString(error_id));
            error = 3;
            break;
        }

        // Launch L1 Kernel function
        dim3 Db = dim3(1);
        dim3 Dg = dim3(1, 1, 1);
        l1_differ<<<Dg, Db>>>(d_a, durationL1, d_indexL1);
        (void)hipDeviceSynchronize();
        error_id = hipGetLastError();
        if (error_id != hipSuccess)
        {
            printf("[L1_L2_DIFF.CUH]: Kernel launch/execution L1 Error: %s\n", hipGetErrorString(error_id));
            error = 5;
            break;
        }
        (void)hipDeviceSynchronize();

        // Launch L2 Kernel function
        l2_differ<<<Dg, Db>>>(d_a, durationL2, d_indexL2);
        (void)hipDeviceSynchronize();
        error_id = hipGetLastError();
        if (error_id != hipSuccess)
        {
            printf("[L1_L2_DIFF.CUH] Kernel launch/execution L2 Error: %s\n", hipGetErrorString(error_id));
            error = 5;
            break;
        }
        (void)hipDeviceSynchronize();

        // Copy results from GPU to Host
        error_id = hipMemcpy((void *)h_timeinfoL1, (void *)durationL1, sizeof(unsigned int) * diffSize, hipMemcpyDeviceToHost);
        if (error_id != hipSuccess)
        {
            printf("[L1_L2_DIFF.CUH]: hipMemcpy durationL1 Error: %s\n", hipGetErrorString(error_id));
            error = 6;
            break;
        }
        error_id = hipMemcpy((void *)h_indexL1, (void *)d_indexL1, sizeof(unsigned int) * diffSize, hipMemcpyDeviceToHost);
        if (error_id != hipSuccess)
        {
            printf("[L1_L2_DIFF.CUH]: hipMemcpy d_indexL1 Error: %s\n", hipGetErrorString(error_id));
            error = 6;
            break;
        }

        error_id = hipMemcpy((void *)h_timeinfoL2, (void *)durationL2, sizeof(unsigned int) * diffSize, hipMemcpyDeviceToHost);
        if (error_id != hipSuccess)
        {
            printf("[L1_L2_DIFF.CUH]: hipMemcpy durationL2 Error: %s\n", hipGetErrorString(error_id));
            error = 6;
            break;
        }
        error_id = hipMemcpy((void *)h_indexL2, (void *)d_indexL2, sizeof(unsigned int) * diffSize, hipMemcpyDeviceToHost);
        if (error_id != hipSuccess)
        {
            printf("[L1_L2_DIFF.CUH]: hipMemcpy d_indexL2 Error: %s\n", hipGetErrorString(error_id));
            error = 6;
            break;
        }

#ifdef IsDebug
        for (int i = 0; i < diffSize; i++)
        {
            fprintf(out, "[%d]: L1=%d, L2=%d\n", h_indexL1[i], h_timeinfoL1[i], h_timeinfoL2[i]);
        }
#endif // IsDebug
        (void)hipDeviceSynchronize();

        double avgL1 = 0.;
        double avgL2 = 0.;
        createOutputFile(diffSize, diffSize, h_indexL1, h_timeinfoL1, &avgL1, nullptr, "L1Differ_");
        createOutputFile(diffSize, diffSize, h_indexL2, h_timeinfoL2, &avgL2, nullptr, "L2Differ_");

        absDistance = abs(avgL2 - avgL1);
        printf("[L1_L2_DIFF.CUH]: Compare average values: L1 %f <<>> L2 %f, compute absolute distance: %f\n", avgL1, avgL2, absDistance);
    } while (false);

    // Free Memory on GPU
    if (d_a != nullptr)
    {
        hipFree(d_a);
    }

    if (d_indexL1 != nullptr)
    {
        hipFree(d_indexL1);
    }

    if (d_indexL2 != nullptr)
    {
        hipFree(d_indexL2);
    }

    if (durationL1 != nullptr)
    {
        hipFree(durationL1);
    }

    if (durationL2 != nullptr)
    {
        hipFree(durationL2);
    }

    // Free Memory on Host
    if (h_a != nullptr)
    {
        free(h_a);
    }

    if (h_indexL1 != nullptr)
    {
        free(h_indexL1);
    }

    if (h_indexL2 != nullptr)
    {
        free(h_indexL2);
    }

    if (h_timeinfoL1 != nullptr)
    {
        free(h_timeinfoL1);
    }

    if (h_timeinfoL2 != nullptr)
    {
        free(h_timeinfoL2);
    }

    if (error != 0)
    {
        printErrorCodeInformation(error);
        exit(error);
    }

    (void)hipDeviceReset();
    return absDistance >= tol;
}

#endif // CUDATEST_L1_L2_DIFF_CUH
