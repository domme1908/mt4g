
#ifndef CUDATEST_C1_LINESIZE
#define CUDATEST_C1_LINESIZE

# include <cstdio>

# include "cuda.h"
# include "../eval.h"
# include "../GPU_resources.hip.cu"

__global__ void c1_linesize (unsigned int upperLimit, unsigned int* linesize);

unsigned int launchC1LineSizeKernelBenchmark(int upperLimit, int* error);

unsigned int measure_C1_LineSize(unsigned int c1SizeBytes) {
    unsigned int c1SizeInts = c1SizeBytes >> 2; // / 4;
    unsigned int limit = 512 >> 2; // / 4;
    int error = 0;
    unsigned int upperLimit = c1SizeInts < limit ? c1SizeInts : limit;

    unsigned int lineSize = 0;
    lineSize = launchC1LineSizeKernelBenchmark((int)upperLimit, &error);
    if (error != 0) {
        printErrorCodeInformation(error);
        exit(error);
    }
    return lineSize;
}

unsigned int launchC1LineSizeKernelBenchmark(int upperLimit, int* error) {
    unsigned int lineSize = 0;
    hipError_t error_id;
    unsigned int *h_lineSize = nullptr, *d_lineSize = nullptr;

    do {
        // Allocation on Host Memory
        h_lineSize = (unsigned int *) malloc(sizeof(unsigned int));
        if (h_lineSize == nullptr) {
            printf("[C1_LINESIZE.CUH]: malloc h_lineSize Error\n");
            *error = 1;
            break;
        }

        // Allocation on GPU Memory
        error_id = hipMalloc((void **) &d_lineSize, sizeof(unsigned int));
        if (error_id != cudaSuccess) {
            printf("[C1_LINESIZE.CUH]: hipMalloc d_lineSize Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }
        hipDeviceSynchronize();

        // Launch Kernel function
        dim3 Db = dim3(1);
        dim3 Dg = dim3(1, 1, 1);
        c1_linesize <<<Dg, Db>>>(upperLimit, d_lineSize);
        hipDeviceSynchronize();

        error_id = hipGetLastError();
        if (error_id != cudaSuccess) {
            printf("[C1_LINESIZE.CUH]: Kernel launch/execution with clock Error: %s\n", hipGetErrorString(error_id));
            *error = 5;
            break;
        }
        hipDeviceSynchronize();

        // Copy results from GPU to Host
        error_id = hipMemcpy((void *) h_lineSize, (void *) d_lineSize, sizeof(unsigned int), hipMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[C1_LINESIZE.CUH]: hipMemcpy d_lineSize Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }
        hipDeviceSynchronize();

        lineSize = h_lineSize[0];
        hipDeviceSynchronize();
    } while(false);

    if (d_lineSize != nullptr) {
        hipFree(d_lineSize);
    }

    if (h_lineSize != nullptr) {
        free(h_lineSize);
    }

    hipDeviceReset();

    return lineSize;
}

__global__ void c1_linesize (unsigned int upperLimit, unsigned int *lineSize) {
    unsigned int start_time, end_time;
    unsigned int j = 0;

    // Using cold cache misses for this cache
    for (int k = 0; k < upperLimit; k++) {
        start_time = clock();
        j = arr[j];
        s_index[k] = j;
        end_time = clock();
        s_tvalue[k] = end_time - start_time;
    }

    unsigned long long ref = (s_tvalue[14] + s_tvalue[15] + s_tvalue[16]) / 3;
    int firstIndexMiss = 16;
    while(s_tvalue[firstIndexMiss] <= ref + 25) { //25 tolerance
        firstIndexMiss++;
    }

    int secondIndexMiss = firstIndexMiss+1;
    while(s_tvalue[secondIndexMiss] < ref + 25) {
        secondIndexMiss++;
    }

    lineSize[0] = ((unsigned int) secondIndexMiss - (unsigned int) firstIndexMiss) * 4;
}

#endif //CUDATEST_C1_LINESIZE

