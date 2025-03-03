
#ifndef CUDATEST_RO
#define CUDATEST_RO

# include <cstdio>
# include <cstdint>

# include "binarySearch.hip.h"

# include "eval.hip.h"
# include "utils.h"
# include "GPU_resources.hip.h"


__global__ void RO_size (const unsigned int* __restrict__ my_array, int array_length,  unsigned int * duration, unsigned int *index, bool* isDisturbed);

bool launchROBenchmark(int N, int stride, double *avgOut, unsigned int* potMissesOut, unsigned int** time, int* error);

CacheSizeResult measure_ReadOnly() {
    int absoluteLowerBoundary = 1024;
    int absoluteUpperBoundary = 1024 << 10; // 1024 * 1024
    int widenBounds = 8;

    int bounds[2] = {absoluteLowerBoundary, absoluteUpperBoundary};
    getBoundaries(launchROBenchmark, bounds, 5);
#ifdef IsDebug
    fprintf(out, "Got Boundaries: %d...%d\n", bounds[0], bounds[1]);
#endif //IsDebug
    printf("Got Boundaries: %d...%d\n", bounds[0], bounds[1]);

    int cp = -1;
    int begin = bounds[0] - widenBounds;
    int end = bounds[1] + widenBounds;
    int stride = 8;
    int arrayIncrease = 1;

    while (cp == -1 && begin >= absoluteLowerBoundary / sizeof(int) - widenBounds && end <= absoluteUpperBoundary / sizeof(int) + widenBounds) {
        cp = wrapBenchmarkLaunch(launchROBenchmark, begin, end, stride, arrayIncrease, "RO");

        if (cp == -1) {
            begin = begin - (end - begin);
            end = end + (end - begin);
#ifdef IsDebug
            fprintf(out, "\nGot Boundaries: %d...%d\n", begin, end);
#endif //IsDebug
            printf("\nGot Boundaries: %d...%d\n", begin, end);
        }
    }

    CacheSizeResult result;
    int cacheSizeInInt = (begin + cp * arrayIncrease);
    result.CacheSize = (cacheSizeInInt << 2); // * 4);
    result.realCP = cp > 0;
    result.maxSizeBenchmarked = end << 2; // * 4;
    return result;
}


bool launchROBenchmark(int N, int stride, double *avgOut, unsigned int* potMissesOut, unsigned int** time, int* error) {
    hipDeviceReset();
    hipError_t error_id;

    unsigned int *h_a = nullptr, *h_index = nullptr, *h_timeinfo = nullptr,*lines = nullptr,
    *d_a = nullptr, *duration = nullptr, *d_index = nullptr;
    bool *disturb = nullptr, *d_disturb = nullptr;

    do {
        // Allocate Memory on Host
        h_a = (unsigned int *) malloc(sizeof(unsigned int) * N);
        if (h_a == nullptr) {
            printf("[RO.CUH]: malloc h_a Error\n");
            *error = 1;
            break;
        }

        h_index = (unsigned int *) malloc(sizeof(unsigned int) * MEASURE_SIZE);
        if (h_index == nullptr) {
            printf("[RO.CUH]: malloc h_index Error\n");
            *error = 1;
            break;
        }

        h_timeinfo = (unsigned int *) malloc(sizeof(unsigned int) * MEASURE_SIZE);
        if (h_timeinfo == nullptr) {
            printf("[RO.CUH]: malloc h_timeinfo Error\n");
            *error = 1;
            break;
        }

        disturb = (bool *) malloc(sizeof(bool));
        if (disturb == nullptr) {
            printf("[RO.CUH]: malloc disturb Error\n");
            *error = 1;
            break;
        }

        // Allocate Memory on GPU
        error_id = hipMalloc((void **) &d_a, sizeof(unsigned int) * N);
        if (error_id != hipSuccess) {
            printf("[RO.CUH]: hipMalloc d_a Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = hipMalloc((void **) &duration, sizeof(unsigned int) * MEASURE_SIZE);
        if (error_id != hipSuccess) {
            printf("[RO.CUH]: hipMalloc duration Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = hipMalloc((void **) &d_index, sizeof(unsigned int) * MEASURE_SIZE);
        if (error_id != hipSuccess) {
            printf("[RO.CUH]: hipMalloc d_index Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = hipMalloc((void **) &d_disturb, sizeof(bool));
        if (error_id != hipSuccess) {
            printf("[RO.CUH]: hipMalloc disturb Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }

        // Initialize p-chase array
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
        if (error_id != hipSuccess) {
            printf("[RO.CUH]: hipMemcpy d_a Error: %s\n", hipGetErrorString(error_id));
            *error = 3;
            break;
        }

        hipDeviceSynchronize();

        // Launch Kernel function
        dim3 Db = dim3(1);
        dim3 Dg = dim3(1, 1, 1);
        RO_size<<<Dg, Db>>>(d_a, N, duration, d_index, d_disturb);

        hipDeviceSynchronize();

        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[RO.CUH]: Kernel launch/execution Error: %s\n", hipGetErrorString(error_id));
            *error = 5;
            break;
        }
        hipDeviceSynchronize();

        // Copy results from GPU to Host
        error_id = hipMemcpy((void *) h_timeinfo, (void *) duration, sizeof(unsigned int) * MEASURE_SIZE,hipMemcpyDeviceToHost);
        if (error_id != hipSuccess) {
            printf("[RO.CUH]: hipMemcpy duration Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = hipMemcpy((void *) h_index, (void *) d_index, sizeof(unsigned int) * MEASURE_SIZE,hipMemcpyDeviceToHost);
        if (error_id != hipSuccess) {
            printf("[RO.CUH]: hipMemcpy d_index Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = hipMemcpy((void *) disturb, (void *) d_disturb, sizeof(bool), hipMemcpyDeviceToHost);
        if (error_id != hipSuccess) {
            printf("[RO.CUH]: hipMemcpy disturb Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }

        hipDeviceSynchronize();

        if (!*disturb)
            createOutputFile(N, MEASURE_SIZE, h_index, h_timeinfo, avgOut, potMissesOut, "RO_");
    } while(false);

    // Free Memory on GPU
    if (d_a != nullptr) {
        hipFree(d_a);
    }

    if (d_index != nullptr) {
        hipFree(d_index);
    }

    if (duration != nullptr) {
        hipFree(duration);
    }

    // Free Memory on Host
    bool ret = false;
    if (disturb != nullptr) {
        ret = *disturb;
        free(disturb);
    }

    if (h_a != nullptr) {
        free(h_a);
    }

    if (h_index != nullptr) {
        free(h_index);
    }

    if (h_timeinfo != nullptr) {
        if (time != nullptr) {
            time[0] = h_timeinfo;
        } else {
            free(h_timeinfo);
        }
    }

    hipDeviceReset();
    return ret;
}

__global__ void RO_size (const unsigned int* __restrict__ my_array, int array_length, unsigned int * duration, unsigned int *index, bool *isDisturbed) {
    unsigned int start_time, end_time;
    unsigned int j = 0;

    bool dist = false;

    for(int k=0; k<MEASURE_SIZE; k++){
        s_index[k] = 0;
        s_tvalue[k] = 0;
    }

    // First round
    for (int k = 0; k < array_length; k++)
        j = __ldg(&my_array[j]);

    // Second round
    for (int k = 0; k < MEASURE_SIZE; k++) {
        start_time = clock();
        j = __ldg(&my_array[j]);
        s_index[k] = j;
        end_time = clock();
        s_tvalue[k] = end_time-start_time;
    }

    /***
     * sequential load
    for (int k = 0; k < array_length; k++)
        j = __ldg(&my_array[k]);

    //second round
    for (int k = 0; k < MEASURE_SIZE; k++) {
        int l = k % array_length;
        start_time = clock();
        j = __ldg(&my_array[l]);
        s_index[k] = j;
        end_time = clock();
        s_tvalue[k] = end_time-start_time;
    }*/

    for(int k=0; k < MEASURE_SIZE; k++){
        if (s_tvalue[k] > 2000) {
            dist = true;
        }
        index[k]= s_index[k];
        duration[k] = s_tvalue[k];

        *isDisturbed = dist;
    }
}


#endif //CUDATEST_RO


