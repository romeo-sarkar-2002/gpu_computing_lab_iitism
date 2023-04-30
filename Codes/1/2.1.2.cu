#include <stdio.h>
__global__ void GPU ()
{
    for (int i = 0; i < 4; i++)
        printf ("GPU> (%d)Course Name: GPU Computing Lab; Name of Experiment: Programs-> Hello world, a Kernel Call and Passing Parameters; Date: %s\n", i + 1, __DATE__);
    return;
}
int main ()
{
    for (int i = 0; i < 4; i++)
        printf ("CPU> (%d)Course Name: GPU Computing Lab; Name of Experiment: Programs-> Hello world, a Kernel Call and Passing Parameters; Date: %s\n", i + 1, __DATE__);
    GPU <<<1, 1>>> ();
    cudaDeviceReset ();
    return 0;
}