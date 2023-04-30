// Lab Exercise 1.2

#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define ROWS 64
#define COLUMNS 64

__global__ void sum (double a[ROWS][COLUMNS], double b[ROWS][COLUMNS], double s[ROWS][COLUMNS]);
void fill_data (void *p);
void display_matrix (void *p);


int main ()
{
    srand (time (NULL));
    // clock_t c = clock ();
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
    

// 3) Sum two matrices using 2D grid with different block sizes
//[
    printf ("   \033[4mgridDim:\033[m         \033[4mblockDim:\033[m      \033[4mtime(s):\033[m\n");
    
    for (int i = 1; i <= 1024; i *= 2)
    {
        int block_x = i, block_y = 1024 / i;
        dim3 block (block_x, block_y, 1);
        dim3 grid ((ROWS + block_x - 1) / block_x, (COLUMNS + block_y - 1) / block_y, 1);
        
    // 6) show the effect of different block sizes
    //[
        printf ("%04d,%04d,%04d    %04d,%04d,%04d ", grid.x, grid.y, grid.z, block.x, block.y, block.z);
        clock_t c = clock ();
        sum <<<grid, block>>> (device_arr_a, device_arr_b, device_arr_c);
        cudaDeviceSynchronize ();
        c = clock () - c;
        printf ("   %5.3f\n", ((float) (c)) / CLOCKS_PER_SEC);
        // time_t t;
        // }
    }
//]


// 4) Transfer Result (Matrix C) from device to host
//[
    cudaMemcpy (host_arr_c, device_arr_c, ROWS * COLUMNS * sizeof (double), cudaMemcpyDeviceToHost);
//]


// 5) Print the result in matrix format
//[
    // std::cout << "matrix_a: " << std::endl;
    // display_matrix (host_arr_a);
    // std::cout << "matrix_b: " << std::endl;
    // display_matrix (host_arr_b);
    // std::cout << "matrix_c: " << std::endl;
    // display_matrix (host_arr_c);
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
    int global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x, global_threadIdy = blockIdx.y * blockDim.y + threadIdx.y;
    // printf ("blockIdx=(%d,%d,%d);threadIdx=(%d,%d,%d)->{%d,%d,%d}\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, global_threadIdx, global_threadIdy, 0);
    if (global_threadIdx < ROWS)
    {
        if (global_threadIdy < COLUMNS)
        {
            for (int i = 0; i < 1024 * 1024 * 2; i++)
            s[global_threadIdx][global_threadIdy] = a[global_threadIdx][global_threadIdy] + b[global_threadIdx][global_threadIdy];
        }
    }
    return;
}

void fill_data (void *p)
{
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