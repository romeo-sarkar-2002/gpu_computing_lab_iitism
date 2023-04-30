#include <stdio.h>
#include <stdlib.h>
#define N (1024*1024)
#define FULL_DATA_SIZE (N*20)
#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if(error != cudaSuccess)\
    {\
        fprintf (stderr,"Error:%s:%d,",__FILE__,__LINE__);\
        fprintf (stderr,"code:%d,reason:%s\n",error,\
        cudaGetErrorString (error));\
        exit (1);\
    }\
}
__global__ void kernel (int  *a, int  *b, int  *c)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
    {
        c[idx] = (a[idx] + b[idx]) / 2.0;
    }
}

int main (void)
{
    cudaDeviceProp prop;
    int whichDevice;
    CHECK (cudaGetDevice (&whichDevice));
    CHECK (cudaGetDeviceProperties (&prop, whichDevice));
    if (!prop.deviceOverlap)
    {
        printf ("Device will not handle overlaps, so no speed up from streams\n");
        return 0;
    }
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaStream_t stream;
    int  *host_a, *host_b, *host_c;
    int  *dev_a, *dev_b, *dev_c;

    //start the timers
    CHECK (cudaEventCreate (&start));
    CHECK (cudaEventCreate (&stop));

    //initialize the stream
    CHECK (cudaStreamCreate (&stream));
    //allocatethememoryontheGPU
    CHECK (cudaMalloc ((void **) &dev_a, N * sizeof (int)));
    CHECK (cudaMalloc ((void **) &dev_b, N * sizeof (int)));
    CHECK (cudaMalloc ((void **) &dev_c, N * sizeof (int)));
        //allocate host locked memory, used to stream
    CHECK (cudaHostAlloc ((void **) &host_a, FULL_DATA_SIZE * sizeof (int), cudaHostAllocDefault));
    CHECK (cudaHostAlloc ((void **) &host_b, FULL_DATA_SIZE * sizeof (int), cudaHostAllocDefault));
    CHECK (cudaHostAlloc ((void **) &host_c, FULL_DATA_SIZE * sizeof (int), cudaHostAllocDefault));

    for (int i = 0;i < FULL_DATA_SIZE;i++)
    {
        host_a[i] = rand ();
        host_b[i] = rand ();
    }
    CHECK (cudaEventRecord (start, 0));
    //now loop over full data,in bite−sized chunks
    for (int i = 0;i < FULL_DATA_SIZE;i += N)
    {
//copythelockedmemorytothedevice,async
        CHECK (cudaMemcpyAsync (dev_a, host_a + i, N * sizeof (int), cudaMemcpyHostToDevice, stream));
        CHECK (cudaMemcpyAsync (dev_b, host_b + i, N * sizeof (int), cudaMemcpyHostToDevice, stream));

        kernel <<<N / 256, 256, 0, stream >>> (dev_a, dev_b, dev_c);

        //copy the data fromdevicetolockedmemory
        CHECK (cudaMemcpyAsync (host_c + i, dev_c,N * sizeof (int),cudaMemcpyDeviceToHost, stream));

    }
    //copy result chunk from locked to full buffer
    CHECK (cudaStreamSynchronize (stream));

    CHECK (cudaEventRecord (stop, 0));

    CHECK (cudaEventSynchronize (stop));
    CHECK (cudaEventElapsedTime (&elapsedTime, start, stop));
    printf ("The single stream with ID %p was created and the total time taken for (data transfer, computation) is % 8.6f ms\n", stream, elapsedTime);
        //clean up the streams and memory
        CHECK (cudaFreeHost (host_a));
    CHECK (cudaFreeHost (host_b));
    CHECK (cudaFreeHost (host_c));
    CHECK (cudaFree (dev_a));
    CHECK (cudaFree (dev_b));
    CHECK (cudaFree (dev_c));
    CHECK (cudaStreamDestroy (stream));

    return 0;
}
