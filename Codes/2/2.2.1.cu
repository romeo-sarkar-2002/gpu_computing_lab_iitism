#include <cuda_runtime.h>
#include <stdio.h>
#define N 1024
__global__ void calcSqOfDiff (double *a, double *b, double *c)
{
    int i = blockIdx.x;
    if (i < N)
    {
        c[i] = a[i] - b[i];
        c[i] *= c[i];
    }
    return;
}

__global__ void calcSq (double *a, double *b)
{
    int i = blockIdx.x;
    if (i < N)
    {
        b[i] = a[i] * a[i];
    }
    return;
}

int main (int argc, char **argv)
{
    double x[N], y[N], z[N];
    double *dev_x, *dev_y, *dev_z;
    // allocate memory on device
    cudaMalloc ((void **) (&dev_x), N * sizeof (double));
    cudaMalloc ((void **) (&dev_y), N * sizeof (double));
    cudaMalloc ((void **) (&dev_z), N * sizeof (double));
    for (int c = 0, i = 1; c < N; c++, i++)
    {
        x[c] = i * i;
        y[c] = 2 * i + 1;
    }

    // Copy data from host to device
    cudaMemcpy (dev_x, x, N * sizeof (double), cudaMemcpyHostToDevice);
    cudaMemcpy (dev_y, y, N * sizeof (double), cudaMemcpyHostToDevice);

    // launch kernel
    calcSqOfDiff <<<N, 1>>> (dev_x, dev_y, dev_z);
    cudaDeviceSynchronize ();
    // wait for kernel to return
    // Copy result from device to host
    cudaMemcpy (z, dev_z, N * sizeof (double), cudaMemcpyDeviceToHost);
    double sumOfSq = 0;
    for (int c = 0; c < N; c++)
    {
        // printf ("%lf\n", z[c]);
        sumOfSq += z[c];
    }
    // printf ("sum of squares: %lf\n", sumOfSq);
    printf ("Distance between x and y is %lf\n", sqrt (sumOfSq));
    // calculating norm of x
    calcSq <<<N, 1>>> (dev_x, dev_z); 
    cudaDeviceSynchronize ();
    cudaMemcpy (z, dev_z, N * sizeof (double), cudaMemcpyDeviceToHost);
    double sumOfSq_x = 0;
    for (int c = 0; c < N; c++)
    {
        // printf ("%lf\n", z[c]);
        sumOfSq_x += z[c];
    }
    printf ("Norm of x: %lf\n", sqrt (sumOfSq_x));
    // calculating norm of y
    calcSq <<<N, 1>>> (dev_y, dev_z); 
    cudaDeviceSynchronize ();
    cudaMemcpy (z, dev_z, N * sizeof (double), cudaMemcpyDeviceToHost);
    double sumOfSq_y = 0;
    for (int c = 0; c < N; c++)
    {
        // printf ("%lf\n", z[c]);
        sumOfSq_y += z[c];
    }
    printf ("Norm of y: %lf\n", sqrt (sumOfSq_y));
    cudaFree (dev_x);
    cudaFree (dev_y);
    cudaFree (dev_z);
    cudaDeviceReset ();
    return 0;
}