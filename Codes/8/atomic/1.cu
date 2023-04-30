#include <stdio.h>
namespace myNamespace
{
    __global__ void add (int *p)
    {
        (*p) += 1;
        return;
    }
    __global__ void atomicAdd (int *p)
    {
        ::atomicAdd (p, 1);
        return;
    }
    __global__ void atomicSub (int *p)
    {
        ::atomicSub (p, 1);
        return;
    }
}
int main ()
{
    int *a = (int *) (malloc (sizeof (int)));
    int *dev_a;
    cudaMalloc (&dev_a, sizeof (float));
    
    *a = 0;
    cudaMemcpy (dev_a, a, sizeof (int), cudaMemcpyHostToDevice);
    printf ("initial value: %d\n", *a);
    myNamespace::add <<< 1, 2 >>> (dev_a);
    cudaDeviceSynchronize ();
    cudaMemcpy (a, dev_a, sizeof (int), cudaMemcpyDeviceToHost);
    printf ("after add: %d\n", *a);
    myNamespace::atomicAdd <<< 1, 2 >>> (dev_a);
    cudaDeviceSynchronize ();
    cudaMemcpy (a, dev_a, sizeof (int), cudaMemcpyDeviceToHost);
    printf ("after atomicAdd: %d\n", *a);
    myNamespace::atomicSub <<< 1, 2 >>> (dev_a);
    cudaDeviceSynchronize ();
    cudaMemcpy (a, dev_a, sizeof (int), cudaMemcpyDeviceToHost);
    printf ("after atomicSub: %d\n", *a);
    return 0;
}