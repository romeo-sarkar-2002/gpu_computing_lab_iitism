#include <stdio.h>
__global__ void print_name ()
{
    for (int i = 0; i < 10; i++)
        printf ("GPU> Romeo Sarkar (%d)\n", i + 1);
    return;
}
int main ()
{
    for (int i = 0; i < 10; i++)
        printf ("CPU> Romeo Sarkar (%d)\n", i + 1);
    print_name <<<1, 1>>> ();
    cudaDeviceReset ();
    return 0;
}