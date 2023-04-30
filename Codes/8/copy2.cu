#include<stdio.h>
#include<stdlib.h>
#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error!=cudaSuccess)\
    {\
        fprintf (stderr,"Error:%s:%d,",__FILE__,__LINE__);\
        fprintf (stderr,"code:%d,reason:%s\n",error, cudaGetErrorString (error));\
        exit (1);\
    }\
}
__global__ void kernel (float *F, double *D)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0)
    {
        *F = 128.1;
        *D = 128.1;
    }
}
int main (int argc, char **argv)
{
    float *deviceF;
    float h_deviceF;
    double *deviceD;
    double h_deviceD;

    float hostF = 128.1;
    double hostD = 128.1;
    CHECK (cudaMalloc ((void **) &deviceF, sizeof (float)));
    CHECK (cudaMalloc ((void **) &deviceD, sizeof (float)));
    kernel <<<1, 32 >>> (deviceF, deviceD);
    CHECK (cudaMemcpy (&h_deviceF, deviceF, sizeof (float),
        cudaMemcpyDeviceToHost));
    CHECK (cudaMemcpy (&h_deviceD, deviceD, sizeof (double),
        cudaMemcpyDeviceToHost));
    printf ("Host single-precision representation of 128.1 = %.20f\n", hostF);
    printf ("Host double-precision representation of 128.1 = %.20f\n", hostD);
    printf ("Device single-precision representation of 128.1 = %.20f\n", h_deviceF);
    printf ("Device double-precision representation of 128.1 = %.20f\n", h_deviceD);
    printf ("Device and host single-precision representation equal?\n\033[31m%s\033[m\n", hostF == h_deviceF ? "yes" : "no");
    printf ("Deviceandhostdouble-precision representation equal?\n\033[31m%s\033[m\n", hostD == h_deviceD ? "yes" : "no");

    return 0;
}
