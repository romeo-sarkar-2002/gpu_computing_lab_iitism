#include <cuda_runtime.h>
#include <stdio.h>
#define N 10
__global__ void VecAddGPU (int *a, int *b, int *c)
{
    int i = blockIdx.x;
    if (i < N)
    {
        c[i] = a[i] + b[i];
    }
    return;
}

int main (int argc, char **argv)
{
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;
    // allocate memory on device
    cudaMalloc ((void **) (&dev_a), N * sizeof (int));
    cudaMalloc ((void **) (&dev_b), N * sizeof (int));
    cudaMalloc ((void **) (&dev_c), N * sizeof (int));
    for (int i = 0; i < N; i++)
    {
        a[i] = -i;
        b[i] = i * i;
    }

    // Copy data from host to device
    cudaMemcpy (dev_a, a, N * sizeof (int), cudaMemcpyHostToDevice);
    cudaMemcpy (dev_b, b, N * sizeof (int), cudaMemcpyHostToDevice);

    // launch kernel
    VecAddGPU <<<N, 1>>> (dev_a, dev_b, dev_c);
    // Copy result from device to host
    cudaMemcpy (c, dev_c, N * sizeof (int), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < N; i++)
    {
        printf ("%d + %d = %d\n", a[i], b[i], c[i]);
    }
    cudaFree (dev_a);
    cudaFree (dev_b);
    cudaFree (dev_c);
    return 0;
}