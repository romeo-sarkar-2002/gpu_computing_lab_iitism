// #include <cuda_runtime.h>
#include <stdio.h>
int main ()
{
    int deviceCount = 0;
    cudaGetDeviceCount (&deviceCount);
    if (deviceCount == 0)
    {
        printf ("There are no available device(s) that support CUDA\n");
    }
    else
    {
        printf ("Detected %d CUDA Capable device(s)\n", deviceCount);
    }
    int dev = 0;
    cudaSetDevice (dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties (&deviceProp, dev);
    printf ("Device: %d: \"%s\"\n", dev, deviceProp.name);
    printf ("    Wrap size:                                      %d\n", deviceProp.warpSize);
    printf ("    Maximum number of threads per multiprocessor:   %d\n", deviceProp.maxThreadsPerMultiProcessor);
    printf ("    Maximum number of threads per block:            %d\n", deviceProp.maxThreadsPerBlock);
    printf ("    Maximum sizes of each dimension of a block:     %d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf ("    Maximum sizes of each dimension of grid:        %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf ("    Maximum memory pitch:                           %zd bytes\n", deviceProp.memPitch);
    cudaDeviceReset ();
    return 0;
}