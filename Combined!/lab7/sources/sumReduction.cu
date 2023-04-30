#include <stdio.h>
#include "../headers/error.cuh"
#include "../headers/macro.cuh"

#define _SHARED_ARR_LEN_ 439U

__global__ void sumKernel (long long *arr, long long *sum, size_t size)
{
    __shared__ long long s_arr[_SHARED_ARR_LEN_];
    unsigned int s;
    if (FLOOR_DIV (size, 2 * _SHARED_ARR_LEN_) == blockIdx.x)
    {
        s = CEIL_DIV (size % (2 * _SHARED_ARR_LEN_), 2);
    }
    else
    {
        s = _SHARED_ARR_LEN_;
    }
    if ((threadIdx.x < s) && ((2 * (threadIdx.x + blockIdx.x * _SHARED_ARR_LEN_)) < size))
    {
        s_arr[threadIdx.x] = arr[2 * (threadIdx.x + blockIdx.x * _SHARED_ARR_LEN_)];
    }
    __syncthreads ();
    if ((threadIdx.x < s) && ((2 * (threadIdx.x + blockIdx.x * _SHARED_ARR_LEN_) + 1) < size))
    {
        s_arr[threadIdx.x] += arr[2 * (threadIdx.x + blockIdx.x * _SHARED_ARR_LEN_) + 1];
    }
    __syncthreads ();
    // now, find the sum of the entire block:
    for (unsigned int stride = 1; stride < s; stride <<= 1)
    {
        if ((threadIdx.x % (stride << 1)) == 0)
        {
            if ((threadIdx.x + stride) < s)
            {
                s_arr[threadIdx.x] += s_arr[threadIdx.x + stride];
            }
        }
        __syncthreads ();
    }
    sum[blockIdx.x] = s_arr[0];
    return;
}
long long sum (long long *arr, size_t size)
{
    // array will be divided into smaller array of size _SHARED_ARR_LEN_
    long long  sum;
    size_t temp_arr_size = size, temp_sum_arr_size = CEIL_DIV (temp_arr_size, 2 * _SHARED_ARR_LEN_);
    long long *dev_temp_arr = NULL, *dev_temp_sum_arr = NULL;
    struct timespec start, stop;
    timespec_get (&start, TIME_UTC);
    cudaMalloc (&dev_temp_arr, sizeof (long long) * temp_arr_size);
    cudaMemcpy (dev_temp_arr, arr, temp_arr_size * sizeof (long long), cudaMemcpyHostToDevice);
    for (; temp_arr_size > 1; temp_arr_size = temp_sum_arr_size, temp_sum_arr_size = CEIL_DIV (temp_arr_size, 2 * _SHARED_ARR_LEN_))
    {
        cudaMalloc (&dev_temp_sum_arr, sizeof (long long) * temp_sum_arr_size);
        sumKernel <<< temp_sum_arr_size, _SHARED_ARR_LEN_ >>> (dev_temp_arr, dev_temp_sum_arr, temp_arr_size);
        cudaDeviceSynchronize ();
        cudaFree (dev_temp_arr);
        dev_temp_arr = dev_temp_sum_arr;
        dev_temp_sum_arr = NULL;
    }
    cudaMemcpy (&sum, dev_temp_arr, sizeof (long long), cudaMemcpyDeviceToHost);
    cudaFree (dev_temp_arr);
    timespec_get (&stop, TIME_UTC);
    printf ("\033[90mtime taken to compute sum: %.9lf secs.\033[m\n", ((double) (stop.tv_nsec - start.tv_nsec) * 1e-9 + ((double) (stop.tv_sec - start.tv_sec))));
    return sum;
}