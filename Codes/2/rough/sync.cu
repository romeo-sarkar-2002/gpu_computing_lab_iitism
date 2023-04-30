#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
__global__ void increment (int *a)
{
    // printf ("*a: %d ", *a);
    (*a) = (*a) + 10;
    
    // printf ("*a: %d ", *a);
    return;
}

int main ()
{
    int a = 0;
    int *dev;
    cudaMalloc (&dev, sizeof (int));
    cudaMemset (dev, 0, sizeof (int));
    increment <<<1, 1024>>> (dev);
    cudaDeviceSynchronize ();
    cudaMemcpy (&a, dev, sizeof (int), cudaMemcpyDeviceToHost);
    printf ("%d\n", a);
    return 0;
}