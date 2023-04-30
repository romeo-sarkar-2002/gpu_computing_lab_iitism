#include<stdio.h>
#include<stdlib.h>

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if(error != cudaSuccess)\
    {\
        fprintf(stderr,"Error:%s:%d, ",__FILE__,__LINE__);\
        fprintf(stderr,"code:%d, reason:%s\n",error, cudaGetErrorString(error));\
        exit(1);\
    }\
}


__global__ void fmad_kernel (double x, double y, double *out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0)
    {
        *out = x * x + y;
        // *out = x * x;
        // *out = *out * y;
    }
    return;
}

double host_fmad_kernel (double x, double y)
{
    return x * x + y;
}
enum sgjlskfj {en1, en2};
int main (int argc, char **argv)
{
    double *d_out, h_out;
    double x = 2.891903;
    double y = -3.980364;
    double host_value = host_fmad_kernel (x, y);
    CHECK (cudaMalloc ((void **) &d_out, sizeof (double)));
    fmad_kernel <<<1, 32 >>> (x, y, d_out);
    CHECK (cudaMemcpy (&h_out, d_out, sizeof (double),
        cudaMemcpyDeviceToHost));
    if (host_value == h_out)
    {
        printf ("The device output the same value as the host.\n");
        printf ("The device output is %.20f and the host output is = %.20f\n", h_out, host_value);
    }
    else
    {
        printf ("The device output and host values are different, (host-device) is = %e.\n", fabs (host_value - h_out));
        printf ("The device output is %.20f and the host output is = %.20f\n", h_out, host_value);
    }
    return 0;
}
