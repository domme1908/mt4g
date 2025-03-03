
#ifndef CUDATEST_C15_LINESIZE
#define CUDATEST_C15_LINESIZE

# include <cstdio>


# include "../eval.hip.h"
# include "../GPU_resources.hip.h"

__global__ void c15_linesize (unsigned int* linesize);

unsigned int launchC15LineSizeKernelBenchmark(int* error);

unsigned int measure_C15_LineSize() {
    int error = 0;

    unsigned int lineSize = 0;
    lineSize = launchC15LineSizeKernelBenchmark(&error);
    if (error != 0) {
        printErrorCodeInformation(error);
        exit(error);
    }
    return lineSize;
}

unsigned int launchC15LineSizeKernelBenchmark(int* error) {
    unsigned int lineSize = 0;
    hipError_t error_id;
    unsigned int *h_lineSize = nullptr, *d_lineSize = nullptr;

    do {
        // Allocation on Host Memory
        h_lineSize = (unsigned int *) malloc(sizeof(unsigned int));
        if (h_lineSize == nullptr) {
            printf("[C15_LINESIZE.CUH]: malloc h_lineSize Error\n");
            *error = 1;
            break;
        }
        // Allocation on GPU Memory
        error_id = hipMalloc((void **) &d_lineSize, sizeof(unsigned int));
        if (error_id != hipSuccess) {
            printf("[C15_LINESIZE.CUH]: hipMalloc d_lineSize Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }
        hipDeviceSynchronize();

        // Launch Kernel function
        dim3 Db = dim3(1);
        dim3 Dg = dim3(1, 1, 1);
        c15_linesize <<<Dg, Db>>>(d_lineSize);
        hipDeviceSynchronize();
        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[C15_LINESIZE.CUH]: Kernel launch/execution with clock Error: %s\n", hipGetErrorString(error_id));
            *error = 5;
            break;
        }
        /* copy results from GPU to CPU */
        hipDeviceSynchronize();
        error_id = hipMemcpy((void *) h_lineSize, (void *) d_lineSize, sizeof(unsigned int), hipMemcpyDeviceToHost);
        if (error_id != hipSuccess) {
            printf("[C15_LINESIZE.CUH]: hipMemcpy d_lineSize Error: %s\n", hipGetErrorString(error_id));
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

__global__ void c15_linesize (unsigned int *linesize) {
    unsigned int start_time, end_time;
    unsigned int j = 0;
    int N = 1200;
    int limit = 256;

    // Using cold cache misses for this cache
    for (int k = 0; k < limit; k++) {
        start_time = clock();
        j = arr[j];
        s_index[k] = j;
        end_time = clock();
        j = j % N;
        s_tvalue[k] = end_time - start_time;
    }

    unsigned long long ref = (s_tvalue[14] + s_tvalue[15] + s_tvalue[16]) / 3;
    int firstIndexTotalMiss = 16;
    while(s_tvalue[firstIndexTotalMiss] < ref + 100 && firstIndexTotalMiss < limit) { //100 tolerance
        firstIndexTotalMiss++;
    }

    int secondIndexTotalMiss = firstIndexTotalMiss + 1;
    while(s_tvalue[secondIndexTotalMiss] < ref + 100 && secondIndexTotalMiss < limit) {
        secondIndexTotalMiss++;
    }

    linesize[0] = ((unsigned int) secondIndexTotalMiss - (unsigned int) firstIndexTotalMiss) * 4;
}

#endif //CUDATEST_C15_LINESIZE
