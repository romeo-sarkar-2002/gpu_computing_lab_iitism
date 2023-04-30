#include <stdio.h>
#include "lib.cuh"

#define N 2

// This global function compute the square of matrix C which is sum of matrix A and B
__global__ void driver(matrix &A, matrix &B, matrix &C){
    add_matrix_GPU(A.device_pointer, B.device_pointer, C.device_pointer, C.rows, C.cols);
}

int main()
{
    srand (234);
    matrix A(N, N), B(N, N), C(N, N), D(N, N);

    A.init_rand();
    printf("Matrix A:\n");
    A.display();

    B.init_rand();
    printf("Matrix B:\n");
    B.display();

    dim3 block(1);
    dim3 grid(N, N);
    driver<<<grid, block>>>(A, B, C);
    C.D2H();
    // mul_matrix_GPU<<<grid, block>>> (C.device_pointer, C.device_pointer, D.device_pointer, C.rows, C.cols, C.cols);
    D.D2H();

    printf("Matrix C:\n");
    C.display();
    printf("Matrix D:\n");
    D.display();

    return 0;
}