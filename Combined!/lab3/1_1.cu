// Lab Exercise 1.1

#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ROWS 4
#define COLUMNS 8


__global__ void sum (double a[ROWS][COLUMNS], double b[ROWS][COLUMNS], double s[ROWS][COLUMNS]);
void fill_data (void *p);
void display_matrix (void *p);


int main ()
{
    srand (time (NULL));

    double (*host_arr_a)[COLUMNS], (*host_arr_b)[COLUMNS], (*host_arr_c)[COLUMNS];
    host_arr_a = (double (*)[COLUMNS]) (malloc (ROWS * COLUMNS * sizeof (double)));
    host_arr_b = (double (*)[COLUMNS]) (malloc (ROWS * COLUMNS * sizeof (double)));
    host_arr_c = (double (*)[COLUMNS]) (malloc (ROWS * COLUMNS * sizeof (double)));
    
    fill_data (host_arr_a);
    fill_data (host_arr_b);
    

// 1) Allocate Device Memory:
//[
    double (*device_arr_a)[COLUMNS], (*device_arr_b)[COLUMNS], (*device_arr_c)[COLUMNS];
    cudaMalloc (&device_arr_a, ROWS * COLUMNS * sizeof (double));
    cudaMalloc (&device_arr_b, ROWS * COLUMNS * sizeof (double));
    cudaMalloc (&device_arr_c, ROWS * COLUMNS * sizeof (double));
//]


// 2) Transfer Data (Matrices A and B) from host to device
//[
    cudaMemcpy (device_arr_a, host_arr_a, ROWS * COLUMNS * sizeof (double), cudaMemcpyHostToDevice);
    cudaMemcpy (device_arr_b, host_arr_b, ROWS * COLUMNS * sizeof (double), cudaMemcpyHostToDevice);
//]
    

// 3) Sum two matrices using 2D grid
//[
    dim3 grid (ROWS, COLUMNS, 1);
    dim3 block (1, 1, 1);
    sum <<<grid, block>>> (device_arr_a, device_arr_b, device_arr_c);
    cudaDeviceSynchronize ();
//]


// 4) Transfer Result (Matrix C) from device to host
//[
    cudaMemcpy (host_arr_c, device_arr_c, ROWS * COLUMNS * sizeof (double), cudaMemcpyDeviceToHost);
//]


// 5) Print the result in matrix format
//[
    std::cout << "matrix_a: " << std::endl;
    display_matrix (host_arr_a);
    std::cout << "matrix_b: " << std::endl;
    display_matrix (host_arr_b);
    std::cout << "matrix_c: " << std::endl;
    display_matrix (host_arr_c);
//]


    cudaFree (device_arr_a);
    cudaFree (device_arr_b);
    cudaFree (device_arr_c);
    
    free (host_arr_a);
    free (host_arr_b);
    free (host_arr_c);

    cudaDeviceReset ();

    return 0;
}

__global__ void sum (double a[ROWS][COLUMNS], double b[ROWS][COLUMNS], double s[ROWS][COLUMNS])
{
    printf ("blockIdx=(%d,%d,%d)\n", blockIdx.x, blockIdx.y, blockIdx.z);
    if (blockIdx.x < ROWS)
    {
        if (blockIdx.y < COLUMNS)
        {
            s[blockIdx.x][blockIdx.y] = a[blockIdx.x][blockIdx.y] + b[blockIdx.x][blockIdx.y];
        }
    }
    return;
}

void fill_data (void *p)
{
    // srand (time (NULL) + clock ());
    double (*mat)[COLUMNS] = (double (*)[COLUMNS]) (p);
    for (size_t i = 0; i < ROWS; i++)
    {
        for (size_t j = 0; j < COLUMNS; j++)
        {
            mat[i][j] = (double) (rand () % 100 - rand () % 100);
        }
    }
    return;
}

void display_matrix (void *p)
{
    double (*mat)[COLUMNS] = (double (*)[COLUMNS]) p;
    for (size_t i = 0; i < ROWS; i++)
    {
        for (size_t j = 0; j < COLUMNS; j++)
        {
            printf ("%7.2f ", mat[i][j]);
        }
        printf ("\n");
    }
}
