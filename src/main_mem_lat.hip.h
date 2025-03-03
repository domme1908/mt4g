
#ifndef CUDATEST_MAINMEM_LAT
#define CUDATEST_MAINMEM_LAT

# include <cstdio>


# include "eval.hip.h"
# include "GPU_resources.hip.h"

__global__ void main_lat (unsigned int * my_array, unsigned int * time);
__global__ void main_lat_globaltimer (unsigned int * my_array, unsigned int * time);

LatencyTuple launchMainLatKernelBenchmark(int N, int stride, int* error);

LatencyTuple measure_main_Lat(int l2SizeInBytes, int stride) {
    int error = 0;
    LatencyTuple lat = launchMainLatKernelBenchmark(l2SizeInBytes, stride, &error);
    if (error != 0) {
        printErrorCodeInformation(error);
        exit(error);
    }
    return lat;
}


LatencyTuple launchMainLatKernelBenchmark(int N, int stride, int* error) {
    LatencyTuple result;
    hipError_t error_id;

    unsigned int* h_a = nullptr, *h_time = nullptr, *d_a = nullptr, *d_time = nullptr;

    do {
        // Allocate Memory on Host
        h_a = (unsigned int *) malloc(sizeof(unsigned int) * (N));
        if (h_a == nullptr) {
            printf("[MAIN_MEM_LAT.CUH]: malloc h_a Error\n");
            *error = 1;
            break;
        }

        h_time = (unsigned int *) malloc(sizeof(unsigned int));
        if (h_time == nullptr) {
            printf("[MAIN_MEM_LAT.CUH]: malloc h_time Error\n");
            *error = 1;
            break;
        }

        // Allocate Memory on GPU
        error_id = hipMalloc((void **) &d_a, sizeof(unsigned int) * (N));
        if (error_id != hipSuccess) {
            printf("[MAIN_MEM_LAT.CUH]: hipMalloc d_a Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = hipMalloc((void **) &d_time, sizeof(unsigned int));
        if (error_id != hipSuccess) {
            printf("[MAIN_MEM_LAT.CUH]: hipMalloc d_time Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }

        // Initialize p-chase array
        for (int i = 0; i < N; i++) {
            //original:
            h_a[i] = (i + stride) % N;
        }

        // Copy array from Host to GPU
        error_id = hipMemcpy(d_a, h_a, sizeof(unsigned int) * N, hipMemcpyHostToDevice);
        if (error_id != hipSuccess) {
            printf("[MAIN_MEM_LAT.CUH]: hipMemcpy h_a Error: %s\n", hipGetErrorString(error_id));
            *error = 3;
            break;
        }
        hipDeviceSynchronize();

        // Launch Kernel function with clock function
        dim3 Db = dim3(1);
        dim3 Dg = dim3(1, 1, 1);
        main_lat <<<Dg, Db>>>(d_a, d_time);

        hipDeviceSynchronize();

        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[MAIN_MEM_LAT.CUH]: Kernel launch/execution with clock Error:%s\n", hipGetErrorString(error_id));
            *error = 5;
            break;
        }
        hipDeviceSynchronize();

        // Copy results from GPU to Host
        error_id = hipMemcpy((void *) h_time, (void *) d_time, sizeof(unsigned int), hipMemcpyDeviceToHost);
        if (error_id != hipSuccess) {
            printf("[MAINMEMTEST.CUH]: hipMemcpy d_time Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }
        hipDeviceSynchronize();

        unsigned int lat = h_time[0];
#ifdef IsDebug
        fprintf(out, "Measured Main avg latencyCycles is %d cycles\n", lat);
#endif //IsDebug
        result.latencyCycles = lat;
        hipDeviceSynchronize();

        // Launch Kernel function with globaltimer
        main_lat_globaltimer<<<Dg, Db>>>(d_a, d_time);

        hipDeviceSynchronize();

        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[MAIN_MEM_LAT.CUH]: Kernel launch/execution with clock Error:%s\n", hipGetErrorString(error_id));
            *error = 5;
            break;
        }
        hipDeviceSynchronize();

        // Copy results from Host to GPU
        error_id = hipMemcpy((void *) h_time, (void *) d_time, sizeof(unsigned int), hipMemcpyDeviceToHost);
        if (error_id != hipSuccess) {
            printf("[MAINMEMTEST.CUH]: hipMemcpy d_time Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }
        hipDeviceSynchronize();

        lat = h_time[0];
#ifdef IsDebug
        fprintf(out, "Measured Main avg latencyCycles is %d nanoseconds\n", lat);
#endif //IsDebug
        result.latencyNano = lat;
    } while(false);

    // Free Memory on GPU
    if (d_a != nullptr) {
        hipFree(d_a);
    }

    if (d_time != nullptr) {
        hipFree(d_time);
    }

    // Free Memory on Host
    if (h_a != nullptr) {
        free(h_a);
    }

    if (h_time != nullptr) {
        free(h_time);
    }

    hipDeviceReset();

    return result;
}

__global__ void main_lat (unsigned int * my_array, unsigned int *time) {
    int iter = 2048;

    unsigned int start_time, end_time;
    unsigned int j = 0;

    // Warming up
    for (int k = 0; k < 32; k++) {
        j = my_array[j];
    }

    // No first round required
    start_time = clock();
    for (int k = 0; k < iter; k++) {
        asm volatile(
            "ld.global.cg.u32 %0, [%1];\n\t" : "=r"(j) : "r"(my_array+j) : "memory"
        );
    }
    s_index[0] = j;
    end_time = clock();

    unsigned int diff = end_time - start_time;

    time[0] = diff / iter;
}

__global__ void main_lat_globaltimer (unsigned int * my_array, unsigned int *time) {
    int iter = 2048;

    unsigned long long start_time, end_time;
    unsigned int j = 0;
    unsigned int start_lo, start_hi, end_lo, end_hi;

    // Warming up
    for (int k = 0; k < 32; k++) {
        j = my_array[j];
    }

    // No first round required
    asm volatile(
        "s_memtime s[0:1]\n\t"
        "s_mov_b32 %0, s0\n\t"
        "s_mov_b32 %1, s1\n\t"
        : "=r"(start_lo), "=r"(start_hi)
        :
        : "s0", "s1"
    );
    start_time = ((unsigned long long)start_hi << 32) | start_lo;
    for (int k = 0; k < iter; k++) {
        asm volatile(
                "ld.global.cg.u32 %0, [%1];\n\t" : "=r"(j) : "r"(my_array+j) : "memory"
                );
    }
    s_index[0] = j;
    asm volatile(
        "s_memtime s[0:1]\n\t"
        "s_mov_b32 %0, s0\n\t"
        "s_mov_b32 %1, s1\n\t"
        : "=r"(end_lo), "=r"(end_hi)
        :
        : "s0", "s1"
    );
    end_time = ((unsigned long long)end_hi << 32) | end_lo;

    unsigned int diff = (unsigned int) (end_time - start_time);

    time[0] = diff / iter;
}

#endif //CUDATEST_MAINMEM_LAT

