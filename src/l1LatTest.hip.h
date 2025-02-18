
#ifndef CUDATEST_L1LATTEST
#define CUDATEST_L1LATTEST

# include <cstdio>

# include "cuda.h"
# include "eval.hip.h"
# include "utils.h"
# include "GPU_resources.hip.h"

__global__ void l1_lat_test (unsigned int * my_array, int array_length, unsigned int * duration, unsigned int* index, bool* isDisturbed);

bool launchL1LatTestKernelBenchmark(int N, int stride, double *avgOut, unsigned int* potMissesOut, unsigned int** time, int* error);

#define FreeMeasureL1LatTest()  \
free(avg);                      \
free(misses);                   \
free(time);                     \

void measure_L1_LatTest() {
    double *avg = (double*) malloc(sizeof(double));
    unsigned int* misses = (unsigned int*) malloc(sizeof(unsigned int));

    unsigned int** time = (unsigned int**) malloc(sizeof(unsigned int*));
    if (avg == nullptr || misses == nullptr || time == nullptr) {
        FreeMeasureL1LatTest()
        printErrorCodeInformation(1);
        exit(1);
    }

    int stride = 8;
    int arrSize = 200;
    int error = 0;
    bool dist = true;
    int count = 5;

    while (dist && count > 0) {
        dist = launchL1LatTestKernelBenchmark(arrSize, stride, avg, misses, time, &error);
        --count;
    }

    free(time[0]);
    FreeMeasureL1LatTest()

    if (error != 0) {
        printErrorCodeInformation(error);
        exit(error);
    }
}

bool launchL1LatTestKernelBenchmark(int N, int stride, double *avgOut, unsigned int* potMissesOut, unsigned int** time, int *error)  {
    hipError_t error_id;

    unsigned int *h_a = nullptr, *h_index = nullptr, *h_timeinfo = nullptr, *d_a = nullptr, *d_index = nullptr, *duration = nullptr,*lines = nullptr;
    bool *disturb = nullptr, *d_disturb = nullptr;

    do {
        // Allocate Memory on Host
        h_a = (unsigned int *) malloc(sizeof(unsigned int) * (N));
        if (h_a == nullptr) {
            printf("[L1LATTEST.CUH]: malloc h_a Error\n");
            *error = 1;
            break;
        }

        h_index = (unsigned int *) malloc(sizeof(unsigned int) * MEASURE_SIZE);
        if (h_index == nullptr) {
            printf("[L1LATTEST.CUH]: malloc h_index Error\n");
            *error = 1;
            break;
        }

        h_timeinfo = (unsigned int *) malloc(sizeof(unsigned int) * MEASURE_SIZE);
        if (h_timeinfo == nullptr) {
            printf("[L1LATTEST.CUH]: malloc h_timeinfo Error\n");
            *error = 1;
            break;
        }

        disturb = (bool *) malloc(sizeof(bool));
        if (disturb == nullptr) {
            printf("[L1LATTEST.CUH]: malloc disturb Error\n");
            *error = 1;
            break;
        }

        // Allocate Memory on GPU
        error_id = hipMalloc((void **) &d_a, sizeof(unsigned int) * (N));
        if (error_id != cudaSuccess) {
            printf("[L1LATTEST.CUH]: hipMalloc d_a Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = hipMalloc((void **) &d_index, sizeof(unsigned int) * MEASURE_SIZE);
        if (error_id != cudaSuccess) {
            printf("[L1LATTEST.CUH]: hipMalloc d_index Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = hipMalloc((void **) &duration, sizeof(unsigned int) * MEASURE_SIZE);
        if (error_id != cudaSuccess) {
            printf("[L1LATTEST.CUH]: hipMalloc duration Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = hipMalloc((void **) &d_disturb, sizeof(bool));
        if (error_id != cudaSuccess) {
            printf("[L1LATTEST.CUH]: hipMalloc d_disturb Error: %s\n", hipGetErrorString(error_id));
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


        // Copy array from GPU to Host
        error_id = hipMemcpy(d_a, h_a, sizeof(unsigned int) * N, hipMemcpyHostToDevice);
        if (error_id != cudaSuccess) {
            printf("[L1LATTEST.CUH]: hipMemcpy d_a Error: %s\n", hipGetErrorString(error_id));
            *error = 3;
            break;
        }
        hipDeviceSynchronize();

        // Launch Kernel function
        dim3 Db = dim3(1);
        dim3 Dg = dim3(1, 1, 1);
        l1_lat_test <<<Dg, Db>>>(d_a, N, duration, d_index, d_disturb);

        hipDeviceSynchronize();
        error_id = hipGetLastError();
        if (error_id != cudaSuccess) {
            printf("[L1LATTEST.CUH]: Kernel launch/execution with clock Error: %s\n", hipGetErrorString(error_id));
            *error = 5;
            break;
        }
        hipDeviceSynchronize();

        // Copy results from GPU to Host
        error_id = hipMemcpy((void *) h_timeinfo, (void *) duration, sizeof(unsigned int) * MEASURE_SIZE,hipMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[L1LATTEST.CUH]: hipMemcpy duration Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = hipMemcpy((void *) h_index, (void *) d_index, sizeof(unsigned int) * MEASURE_SIZE,hipMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[L1LATTEST.CUH]: hipMemcpy d_index Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = hipMemcpy((void *) disturb, (void *) d_disturb, sizeof(bool), hipMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[L1LATTEST.CUH]: hipMemcpy d_disturb Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }
        hipDeviceSynchronize();

        createOutputFile(N, MEASURE_SIZE, h_index, h_timeinfo, avgOut, potMissesOut, "L1Lat_");
    } while(false);

    // Free Memory on GPU
    if (d_a != nullptr) {
        hipFree(d_a);
    }

    if (duration != nullptr) {
        hipFree(duration);
    }

    if (d_index != nullptr) {
        hipFree(d_index);
    }

    if (d_disturb != nullptr) {
        hipFree(d_disturb);
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

    if (h_timeinfo != nullptr) {
        if (time != nullptr) {
            time[0] = h_timeinfo;
        } else {
            free(h_timeinfo);
        }
    }

    if (h_index != nullptr) {
        free(h_index);
    }

    hipDeviceReset();
    return ret;
}

__global__ void l1_lat_test (unsigned int * my_array, int array_length, unsigned int * duration, unsigned int* index, bool* isDisturbed) {
    unsigned int start_time, end_time;
    bool dist = false;
    unsigned int j = 0;

    for(int k=0; k<MEASURE_SIZE; k++){
        s_index[k] = 0;
        s_tvalue[k] = 0;
    }
    unsigned int* ptr;
	for (int k = 0; k < array_length; k++) {
        ptr = my_array + j;
        asm volatile ("ld.global.ca.u32 %0, [%1];" : "=r"(j) : "l"(ptr) : "memory");
    }

    for (int k = 0; k < MEASURE_SIZE; k++) {
        ptr = my_array + j;
        start_time = clock();
        asm volatile ("ld.global.ca.u32 %0, [%1];" : "=r"(j) : "l"(ptr) : "memory");
        s_index[k] = j;
        end_time = clock();
        s_tvalue[k] = end_time - start_time;
    }

    for(int k=0; k<MEASURE_SIZE; k++){
        if (s_tvalue[k] > 1200) {
            //printf("boom\n");
            dist = true;
        }
        index[k]= s_index[k];
        duration[k] = s_tvalue[k];
    }

    *isDisturbed = dist;
}

#endif //CUDATEST_L1LATTEST
