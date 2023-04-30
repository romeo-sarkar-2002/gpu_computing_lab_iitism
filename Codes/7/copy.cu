#include <cuda_runtime.h>
#include <stdio.h>
#define N 100
#define BD 256

#define CHECK(call) \
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf (stderr, "error: %s: %d,", __FILE__, __LINE__);\
        fprintf (stderr, "code:%d, reason:%s\n", error, cudaGetErrorString (error));\
        exit (1);\
    }\
}

__global__ void sumReduce (float *dev_a, float *dev_b)
{
    __shared__ float partialSum[BD];
    partialSum[threadIdx.x] = dev_a[blockIdx.x * blockDim.x + threadIdx.x];
    unsigned int t = threadIdx.x;
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        __syncthreads ();
        if ((t % (2 * stride)) == 0)
        {
            partialSum[t] += partialSum[t + stride];
        }
    }
    if (0 == threadIdx.x)
    {
        dev_b[blockIdx.x] = partialSum[0];
    }
    return;
}

int main (int argc, char **argv)
{
    float a[N], b[N];
    float *dev_a, *dev_b;
    int bdimx = BD;
    float elapsedTime;
    dim3 block (bdimx);
    dim3 grid ((N + block.x - 1) / block.x, 1, 1);
    cudaEvent_t start, stop;
    CHECK (cudaEventCreate (&start));
    CHECK (cudaEventCreate (&stop));
    printf ("Array Size is = %d\n", N);
    //allocate the memory on device
    CHECK (cudaMalloc ((void **) &dev_a, N * sizeof (float)));
    CHECK (cudaMalloc ((void **) &dev_b, N * sizeof (float)));
    for (int i = 0; i < N; i++) 
    {
        a[i] = 1;
        // a[i] = i + 1;
        // a[i] = ((float) (rand ())) / (float) (RAND_MAX);
    }
    //Cuda events for time measure
    CHECK (cudaEventRecord (start, 0));
    cudaMemcpy (dev_a, a, N * sizeof (float), cudaMemcpyHostToDevice);
    CHECK (cudaEventRecord (stop, 0));
    CHECK (cudaEventSynchronize (stop));
    cudaEventElapsedTime (&elapsedTime, start, stop);
    printf ("Time to do memory transfer of array a from host to device is %3.6f ms\n", elapsedTime);

    //kernel launch
    CHECK (cudaEventRecord (start, 0));
    sumReduce <<<grid, block>>> (dev_a, dev_b);
   //Copy result from device to host
    CHECK (cudaMemcpy (b, dev_b, N * sizeof (float), cudaMemcpyDeviceToHost));
    CHECK (cudaEventRecord (stop, 0));
    CHECK (cudaEventSynchronize (stop));
    cudaEventElapsedTime (&elapsedTime, start, stop);
    printf ("Time to do sum reduction is %3.6f ms\n", elapsedTime);
    printf ("Sum = %f\n", b[0]);
    cudaDeviceSynchronize ();
    cudaEventDestroy (start);
    cudaEventDestroy (stop);
    cudaFree (dev_a);
    cudaFree (dev_b);
    return 0;
}