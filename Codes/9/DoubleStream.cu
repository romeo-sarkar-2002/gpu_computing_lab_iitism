#include <stdio.h>
#include <cuda_runtime.h>
#include "Error.cuh"

#define floorDiv(a, b) ((a) / (b))
#define ceilDiv(a, b) (((a) + (b) - 1) / (b))

#define BLOCK_SIZE 131
#define MAX_DEVICE_MEMORY_SIZE 1024 * 1024 * 256 // -> 256 MB
/** since there are two streams, we will copy a maximum of half of this size per stream */

const size_t SHARED_MEM_SIZE = 1024 * sizeof (long long); // -> 16 KB
/** here we are sure that shared memory won't interfere with the block size
 * since max block size is 1024 (x * y * z).
 * hence shared memory will always be available for any block config.
*/

#define STREAMS 2
const size_t maxMemoryPerStream = MAX_DEVICE_MEMORY_SIZE / STREAMS;

__global__ void maxKernel (long long *arr, size_t siz, long long *tmp)
{
    const size_t s_arr_siz = SHARED_MEM_SIZE / sizeof (long long);
    __shared__ long long s_arr[s_arr_siz];
    if (2 * (blockIdx.x * blockDim.x + threadIdx.x) < siz)
    {
        if (2 * (blockIdx.x * blockDim.x + threadIdx.x) + 1 < siz)
        {
            if (arr[2 * (blockIdx.x * blockDim.x + threadIdx.x)] > arr[2 * (blockIdx.x * blockDim.x + threadIdx.x) + 1])
            {
                s_arr[threadIdx.x] = arr[2 * (blockIdx.x * blockDim.x + threadIdx.x)];
            }
            else
            {
                s_arr[threadIdx.x] = arr[2 * (blockIdx.x * blockDim.x + threadIdx.x) + 1];
            }
        }
        else
        {
            s_arr[threadIdx.x] = arr[2 * (blockIdx.x * blockDim.x + threadIdx.x)];
        }
    }
    __syncthreads ();
    size_t arr_size;
    if (blockIdx.x == floorDiv (siz, blockDim.x))
    {
        arr_size = ceilDiv (siz, 2) % blockDim.x;
    }
    else
    {
        arr_size = blockDim.x;
    }
    __syncthreads ();
    for (size_t trailing_stride = arr_size, stride = ceilDiv (arr_size, 2); trailing_stride > 1; trailing_stride = stride, stride = ceilDiv (stride, 2))
    {
        __syncthreads ();
        if (threadIdx.x < stride)
        {
            if (threadIdx.x + stride < trailing_stride)
            {
                if (s_arr[threadIdx.x] < s_arr[threadIdx.x + stride])
                {
                    s_arr[threadIdx.x] = s_arr[threadIdx.x + stride];
                }
            }
        }
    }
    if (0 == threadIdx.x)
    {
        tmp[blockIdx.x] = s_arr[0];
    }
}


__global__ void sumKernel (long long *arr, size_t siz, long long *tmp)
{
    const size_t s_arr_siz = SHARED_MEM_SIZE / sizeof (long long);
    __shared__ long long s_arr[s_arr_siz];
    if (2 * (blockIdx.x * blockDim.x + threadIdx.x) < siz)
    {
        s_arr[threadIdx.x] = arr[2 * (blockIdx.x * blockDim.x + threadIdx.x)];
        if (2 * (blockIdx.x * blockDim.x + threadIdx.x) + 1 < siz)
        {
            s_arr[threadIdx.x] = arr[2 * (blockIdx.x * blockDim.x + threadIdx.x) + 1];
        }
    }
    size_t arr_size;
    if (blockIdx.x == floorDiv (siz, blockDim.x))
    {
        arr_size = ceilDiv (siz, 2) % blockDim.x;
    }
    else
    {
        arr_size = blockDim.x;
    }
    for (size_t trailing_stride = arr_size, stride = ceilDiv (arr_size, 2); trailing_stride > 1; trailing_stride = stride, stride = ceilDiv (stride, 2))
    {
        __syncthreads ();
        if (threadIdx.x < stride)
        {
            if (threadIdx.x + stride < trailing_stride)
            {
                if (s_arr[threadIdx.x] < s_arr[threadIdx.x + stride])
                {
                    s_arr[threadIdx.x] += s_arr[threadIdx.x + stride];
                }
            }
        }
    }
    if (0 == threadIdx.x)
    {
        tmp[blockIdx.x] = s_arr[0];
    }
    return;
}
/** useful global variables */

// long long *arr;
// long long *dev_arr1, *dev_arr2;
// size_t arr_siz = 1024 * 1024 * 128;


// const size_t maxArrSizePerStream = maxMemorySizePerStream / sizeof (long long);

// cudaStream_t stream1, stream2;
// const size_t arrSizePerStream = floorDiv (maxArrSizePerStream * BLOCK_SIZE, BLOCK_SIZE + 1);

const size_t blockSiz = BLOCK_SIZE;

long long computeSum (long long *_arr, size_t _size, cudaStream_t *_stream)
{
    long long *sum;
    cudaHostAlloc (&sum, sizeof (long long), cudaHostAllocDefault);

    long long *d_arr, *d_tmparr;
    // size_t d_arr_siz = floorDiv ((maxMemoryPerStream * blockSiz), blockSiz + 1);
    size_t d_arr_siz = _size;
    size_t d_tmparr_siz = ceilDiv (_size, blockSiz);

    cudaMallocAsync (&d_arr, d_arr_siz, *_stream);
    cudaMallocAsync (&d_tmparr, d_tmparr_siz, *_stream);
    
    cudaMemcpyAsync (&d_arr, _arr, d_arr_siz * sizeof (long long), cudaMemcpyHostToDevice, *_stream);

    for (; d_arr_siz != 1; )
    {
        sumKernel <<< d_tmparr_siz, blockSiz, 0, (*_stream) >>> (d_arr, d_arr_siz, d_tmparr);
        long long *tmp = d_arr;
        d_arr = d_tmparr;
        d_tmparr = tmp;
        d_arr_siz = d_tmparr_siz;
        d_tmparr_siz = ceilDiv (d_tmparr_siz, blockSiz);
    }

    cudaMemcpyAsync (&sum, d_tmparr, sizeof (long long), cudaMemcpyDeviceToHost, *_stream);
    
    cudaFreeAsync (d_arr, *_stream);
    cudaFreeAsync (d_tmparr, *_stream);
    
    long long s = *sum;

    cudaFreeHost (sum);

    printf ("sum: %lld\n", *sum);
    return *sum;
}



void eval ()
{
    // printf ("array size: %zd bytes (%.6f GB)\n", arr_siz * sizeof (long long), ((float (arr_siz)) / (1024 * 1024 * 1024)) * sizeof (long long));

    // cudaHostAlloc (&arr, arr_siz * sizeof (long long), cudaHostAllocDefault);

    // cudaMalloc (&dev_arr1, dev_arr_siz * sizeof (long long));
    // cudaMalloc (&dev_arr2, dev_arr_siz * sizeof (long long));


    // cudaStreamCreate (&stream1);
    // cudaStreamCreate (&stream2);

}
void init (long long *arr, size_t siz)
{
    for (size_t i = 0; i < siz; i++)
    {
        arr[i] = 1;
    }
    return;
}
int main ()
{
    long long *arr;
    size_t arr_siz = 1024 * 1024;
    cudaHostAlloc (&arr, arr_siz * sizeof (long long), cudaHostAllocDefault);
    init (arr, arr_siz);
    cudaStream_t defaultStream = NULL;
    computeSum (arr, arr_siz, &defaultStream);
    cudaStreamSynchronize (defaultStream);
    
    return 0;
}