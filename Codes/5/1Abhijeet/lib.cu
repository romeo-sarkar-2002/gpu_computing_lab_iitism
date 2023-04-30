#include <stdio.h>
#include <cuda_runtime.h>
#include "lib.cuh"

// macros:
#define widthField 8
#define precisionField 0

matrix::matrix() : rows(0), cols(0), device_pointer(NULL), host_pointer(NULL){};
matrix::matrix(int r, int c){
	rows = r;
	cols = c;
	memAllocInBoth();
}
matrix::~matrix()
{
    printf ("\033[33m%p%p\033[m\n", device_pointer, host_pointer);
	if (device_pointer != NULL)
		cudaFree(device_pointer);
	if (host_pointer != NULL)
		free(host_pointer);
}
void matrix::memAllocInBoth()
{
    host_pointer = (double *)malloc(rows * cols * sizeof(double));
    cudaMalloc(&device_pointer, rows * cols * sizeof(double));
}
void matrix::display()
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
            printf("%*.*lf", widthField, precisionField, host_pointer[i * cols + j]);
        printf("\n");
    }
}
void matrix::init_rand()
{
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			host_pointer[i * cols + j] = rand() % 10 - 4;
	H2D();
}
void matrix::H2D(){cudaMemcpy(device_pointer, host_pointer, rows * cols * sizeof(double), cudaMemcpyHostToDevice);}
void matrix::D2H(){cudaMemcpy(host_pointer, device_pointer, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);}

__device__ void add_matrix_GPU(double *d_MatA, double *d_MatB, double *d_MatC, int rows, int cols){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if (i < rows && j < cols)
		d_MatC[i * cols + j] = d_MatA[i * cols + j] + d_MatB[i * cols + j];
}

__global__ void mul_matrix_GPU(double *d_MatA, double *d_MatB, double *d_MatC, int rows, int width, int cols){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < rows && j < cols){
        double dotp = 0;
        for (int k = 0; k < width; k++)
            dotp += d_MatA[i * width + k] * d_MatB[k * cols + j];
        d_MatC[i * cols + j] = dotp;
    }
}