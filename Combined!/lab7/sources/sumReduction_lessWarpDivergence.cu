#include <stdio.h>
#include "../headers/macro.cuh"
#define BLOCK_SIZE 473U

__global__ void sumKernel_lessWarpDivergence (long long *arr, long long *sum, unsigned int size)
{
    __shared__ long long s_arr[BLOCK_SIZE];
    unsigned int globalIdx = threadIdx.x + blockIdx.x * BLOCK_SIZE;
    if (globalIdx < size)
    {
        s_arr[threadIdx.x] = arr[globalIdx];
    }
    __syncthreads ();
    unsigned int trailing_stride, stride;
    if (FLOOR_DIV (size, BLOCK_SIZE) == blockIdx.x)
        trailing_stride = size % BLOCK_SIZE;
    else
        trailing_stride = BLOCK_SIZE;
    stride = CEIL_DIV (trailing_stride, 2);
    for (; trailing_stride > 1; trailing_stride = stride, stride = CEIL_DIV (stride, 2))
    {
        if (threadIdx.x < stride)
        {
            if ((threadIdx.x + stride) < trailing_stride)
            {
                s_arr[threadIdx.x] += s_arr[threadIdx.x + stride];
            }
        }
        else
        {
            goto finish_line;
        }
        __syncthreads ();
    }
    if (0 == threadIdx.x)
    {
        sum[blockIdx.x] = s_arr[0];
    }
    finish_line:
    return;
}
long long sum_lessWarpDivergence (long long *arr, size_t size)
{
    long long sum;
    size_t temp_arr_size = size, temp_sum_arr_size = CEIL_DIV (size, BLOCK_SIZE);
    long long *dev_temp_arr = NULL, *dev_temp_sum_arr = NULL;
    struct timespec start, stop;
    timespec_get (&start, TIME_UTC);
    cudaMalloc (&dev_temp_arr, sizeof (long long) * temp_arr_size);
    cudaMemcpy (dev_temp_arr, arr, temp_arr_size * sizeof (long long), cudaMemcpyHostToDevice);
    for (; temp_arr_size > 1; temp_arr_size = temp_sum_arr_size, temp_sum_arr_size = CEIL_DIV (temp_sum_arr_size, BLOCK_SIZE))
    {
        cudaMalloc (&dev_temp_sum_arr, sizeof (long long) * temp_sum_arr_size);
        sumKernel_lessWarpDivergence <<< temp_sum_arr_size, BLOCK_SIZE >>> (dev_temp_arr, dev_temp_sum_arr, temp_arr_size);
        cudaDeviceSynchronize ();
        cudaFree (dev_temp_arr);
        dev_temp_arr = dev_temp_sum_arr;
        dev_temp_sum_arr = NULL; 
    }
    cudaMemcpy (&sum, dev_temp_arr, sizeof (long long), cudaMemcpyDeviceToHost);
    cudaFree (dev_temp_arr);
    timespec_get (&stop, TIME_UTC);
    printf ("\033[90mtime taken to compute sum_lessWarpDivergence: %.9lf secs.\033[m\n", ((double) (stop.tv_nsec - start.tv_nsec) * 1e-9 + ((double) (stop.tv_sec - start.tv_sec))));
    return sum;
}