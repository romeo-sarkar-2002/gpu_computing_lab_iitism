#include <cuda_runtime.h>
#include <stdio.h>
#define N 1024

__global__ void findSqOfDiff (double *a, double *b, double mean)
{
    int i = blockIdx.x;
    if (i < N)
    {
        b[i] = mean - a[i];
        b[i] *= b[i];
    }
    return;
}

int main (int argc, char **argv)
{
    double arr[N], sqOfDiffFromMean[N];
    double *dev_a1, *dev_a2;
    cudaMalloc (&dev_a1, N * sizeof (double));
    cudaMalloc (&dev_a2, N * sizeof (double));
    // cudaMalloc ()
    double mean = 0;
    for (int c = 0, i = 1; c < N; c++, i++)
    {
        arr[c] = (2 * i + 1);
        mean += arr[c];
    }
    mean /= N;
    printf ("Mean: %lf\n", mean);
    cudaMemcpy (dev_a1, arr, N * sizeof (double), cudaMemcpyHostToDevice);
    
    findSqOfDiff <<<N, 1>>> (dev_a1, dev_a2, mean);
    cudaDeviceSynchronize ();
    cudaMemcpy (sqOfDiffFromMean, dev_a2, N * sizeof (double), cudaMemcpyDeviceToHost);
    double s = 0;
    for (int i = 0; i < N; i++)
    {
        // printf ("%lf\n", sqOfDiffFromMean[i]);
        s += sqOfDiffFromMean[i];
    }
    s /= N;
    printf ("Standard Deviation: %lf\n", sqrt (s));
    cudaFree (dev_a1);
    cudaFree (dev_a2);
    cudaDeviceReset ();
    return 0;
}