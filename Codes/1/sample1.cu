#include <stdio.h>
__global__ void helloFromGPU ()
{
    printf ("Hello World from GPU!\n");
    return;
}
int main ()
{
    printf ("Hello World from CPU!\n");
    helloFromGPU <<<1, 5>>> ();
    cudaDeviceReset ();
    return 0;
}