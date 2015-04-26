#ifndef __CUDACC__
#define __CUDACC__
#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <iostream>

#define n 10
#define W 100

cudaError_t knapsackCuda(int *c, const int *a, const int *b, unsigned int size);

__device__ int maxi(int a, int b) { 
	return (a > b)? a : b; 
}

__global__ void knapsackKernel(int *wt, int *val, int *output, int i) {
	int w = threadIdx.x;

	__syncthreads();
	if (i == 0 || w == 0)
		output[(i*W)+w] = 0;
	else if (wt[i-1] <= w)
		output[(i*W)+w] = maxi(val[i-1] + output[((i-1)*W)+(w-wt[i-1])],  output[((i-1)*W)+w]);
	else
		output[(i*W)+w] = output[((i-1)*W)+w];
	__syncthreads();
   
}

int main() {
    const int val[] = { 60, 100, 120, 80, 90, 110, 70, 50, 130, 40 };
    const int wt[] = { 10, 20, 30, 40, 10, 20, 30, 40, 10, 30 };
	int *output = 0;

	output = (int *)malloc((n+1)*(W+1)*sizeof(int));

    cudaError_t cudaStatus = knapsackCuda(output, val, wt, n);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "knapsackCuda failed!");
        return 1;
    }

	/*for (int i = 0; i <= n; i++)
		for (int j = 0; j <= W; j++) {
			std::cout << output[(i*W) + j] << " ";
			if (j == W)
				std::cout << std::endl;
	}*/

	std::cout << "Maxmimum Value possible for knapsack with capacity " << W << " is : " << output[(n+1)*W] << std::endl;

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

cudaError_t knapsackCuda(int *output, const int *val, const int *wt, unsigned int size) {
    int *dev_val = 0;
    int *dev_wt = 0;
    int *dev_output = 0;
	int i = 0;
    cudaError_t cudaStatus;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_output, (size + 1) * (W + 1) * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc 1 failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_val, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc 2 failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_wt, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc 3 failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_val, val, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy 1 failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_wt, wt, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy 2 failed!");
        goto Error;
    }

	cudaEventRecord(start);
	while (i <= n) {
		knapsackKernel<<<1, W + 1>>>(dev_wt, dev_val, dev_output, i);
		i++;
	}
	cudaEventRecord(stop);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "knapsackKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching knapsackKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(output, dev_output, (size + 1) * (W + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy 4 failed!");
        goto Error;
    }

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	std::cout << "Execution Time : " << milliseconds / 1000 << " seconds" << std::endl;

Error:
    cudaFree(dev_output);
    cudaFree(dev_val);
    cudaFree(dev_wt);
    
    return cudaStatus;
}
