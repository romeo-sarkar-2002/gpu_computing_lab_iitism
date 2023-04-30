#include <stdio.h>
#include <cuda_runtime.h>
#include "Matrix.cuh"
// macros:
// #define widthField 0
#define precisionField 0
#define SHOW_FUNCTION_CALLS 1

// struct Matrix
// {
//     int rows, cols;
//     double *device_pointer, *host_pointer;
    // int flag = 0;

Matrix :: Matrix () : rows (0), cols (0), device_pointer (NULL), host_pointer (NULL)
{
    return;
}
Matrix :: Matrix (int r, int c) : Matrix ()
{
    rows = r;
    cols = c;
    alloc ();
    return;
}
Matrix :: Matrix (const Matrix &M)
{
    #if SHOW_FUNCTION_CALLS == 1
    printf ("\033[90mMatrix (const Matrix &M)\033[m\n");
    #endif
    rows = M.rows;
    cols = M.cols;
    cudaMalloc (&device_pointer, rows * cols * sizeof (double));
    cudaMemcpy (device_pointer, M.device_pointer, rows * cols * sizeof (double), cudaMemcpyDeviceToDevice);
    host_pointer = (double *) (malloc (rows * cols * sizeof (double)));
    memcpy (host_pointer, M.host_pointer, rows * cols * sizeof (double));
    return;
}
Matrix :: Matrix (Matrix &&M)
{
    #if SHOW_FUNCTION_CALLS == 1
    printf ("\033[90mMatrix (Matrix &&M)\033[m\n");
    #endif
    rows = M.rows;
    cols = M.cols;
    device_pointer = M.device_pointer;
    host_pointer = M.host_pointer;
    M.rows = M.cols = 0;
    M.device_pointer = M.host_pointer = NULL;
    return;
}
Matrix Matrix :: operator = (Matrix &M)
{
    #if SHOW_FUNCTION_CALLS == 1
    printf ("\033[90mMatrix operator = (Matrix &M)\033[m\n");
    #endif
    clear ();
    rows = M.rows;
    cols = M.cols;
    cudaMalloc (&device_pointer, rows * cols * sizeof (double));
    cudaMemcpy (device_pointer, M.device_pointer, rows * cols * sizeof (double), cudaMemcpyDeviceToDevice);
    host_pointer = (double *) (malloc (rows * cols * sizeof (double)));
    memcpy (host_pointer, M.host_pointer, rows * cols * sizeof (double));
    return *this;
}
Matrix Matrix :: operator = (Matrix &&M)
{
    #if SHOW_FUNCTION_CALLS == 1
    printf ("\033[90mMatrix operator = (Matrix &&M)\033[m\n");
    #endif
    rows = M.rows;
    cols = M.cols;
    device_pointer = M.device_pointer;
    host_pointer = M.host_pointer;
    M.rows = M.cols = 0;
    M.device_pointer = M.host_pointer = NULL;
    return *this;
}
Matrix :: ~Matrix ()
{
    #if SHOW_FUNCTION_CALLS == 1
    printf ("\033[90m~Matrix () : %p, %p\033[m\n", device_pointer, host_pointer);
    #endif
    if (NULL != device_pointer)
    {
        cudaFree (device_pointer);
    }
    if (NULL != host_pointer)
    {
        free (host_pointer);
    }
    rows = cols = 0;
    device_pointer = host_pointer = NULL;
    return;
}
void Matrix :: alloc ()
{
    cudaMalloc (&device_pointer, rows * cols * sizeof (double));
    host_pointer = (double *) (malloc (rows * cols * sizeof (double)));
    // printf ("hello");
    return;
}
void Matrix :: clear ()
{
    // printf ("%p, %p\n", device_pointer, host_pointer);
    if (NULL != device_pointer)
    {
        cudaFree (device_pointer);
    }
    if (NULL != host_pointer)
    {
        free (host_pointer);
    }
    rows = cols = 0;
    device_pointer = host_pointer = NULL;
    return;
}
// display works on host matrix
// void display ()
// {
//     for (int i = 0; i < rows; i++)
//     {
//         for (int j = 0; j < cols; j++)
//         {
//             printf ("%*.*lf ", widthField, precisionField, host_pointer[i * cols + j]);
//         }
//         printf ("\n");
//     }
//     return;
// }
void Matrix :: display ()
{
    if (NULL == host_pointer)
    {
        #if WARNINGS == 1
        printf ("\nIn function \'\e[33mprint_matrix_yu\e[m\':\n\e[35mwarning:\e[m \'m\' is (null)\n");
        #endif
        return;
    }
    #define BUFFER_SIZE 128
    // double (*mat)[cols] = (double (*)[cols]) (host_pointer);
    int *max_width_arr = (int *) (malloc (cols * sizeof (int)));
    char **mat_of_strs = (char **) malloc (rows * cols * sizeof (char *));
    // char *(*matrix_of_strings)[c] = mat_of_strs;
    char *str;
    int width;
    for (size_t i = 0; i < cols; i++)
    {
        max_width_arr[i] = 1;
        for (size_t j = 0; j < rows; j++)
        {
            str = (char *) malloc (BUFFER_SIZE * sizeof (char));
            width = snprintf (str, BUFFER_SIZE, "%.*lf", precisionField, host_pointer[j * cols + i]);
            str = (char *) realloc (str, ((size_t) (width + 1)) * sizeof (char));
            mat_of_strs[j * cols + i] = str;
            if (max_width_arr[i] < width)
                max_width_arr[i] = width;
        }
    }
    for (size_t i = 0; i < rows; i++)
    {
        printf ("\033[1;32m\xb3\033[m");
        for (size_t j = 0; j < cols; j++)
        {
            width = strlen (mat_of_strs[i * cols + j]);
            for (int x = 0; x < max_width_arr[j] - width; x++)
                printf (" ");
            printf ("%s", mat_of_strs[i * cols + j]);
            if (j != (cols - 1))
                printf (" ");
        }
        printf ("\033[1;32m\xb3\033[m");
        // newline:
        printf ("\n");
    }
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            free (mat_of_strs[i * cols + j]);
    free (mat_of_strs);
    free (max_width_arr);
    return;
}
// void init ()
// {
//     dim3 block (1, 1, 1);
//     dim3 grid (rows, cols, 1);
//     init_GPU <<<grid, block>>> (device_pointer, rows, cols);
//     cudaDeviceSynchronize ();
//     // printf ("\033[31mhere\033[m");
//     D2H ();
//     // printf ("here");
//     return;
// }
void Matrix :: init ()
{
    ::init (host_pointer, rows, cols);
    // cudaDeviceSynchronize ();
    // printf ("\033[31mhere\033[m");
    H2D ();
    // printf ("here");
    return;
}
void Matrix :: H2D ()
{
    cudaMemcpy (device_pointer, host_pointer, cols * rows * sizeof (double), cudaMemcpyHostToDevice);
    return;
}
void Matrix :: D2H ()
{
    cudaMemcpy (host_pointer, device_pointer, cols * rows * sizeof (double), cudaMemcpyDeviceToHost);
    return;
}
Matrix Matrix :: operator + (const Matrix &M)
{
    if (rows != M.rows && cols != M.cols)
    {
        printf ("Matrix1 (%dX%d); Matrix2 (%dX%d)\n", rows, cols, M.rows, M.cols);
        return Matrix ();
    }
    Matrix p (rows, M.cols);
    dim3 block (1, 1, 1);
    dim3 grid (rows, M.cols, 1);
    add_GPU <<< block, grid>>> (device_pointer, M.device_pointer, p.device_pointer, rows, cols);
    cudaDeviceSynchronize ();
    p.D2H ();
    // p.display ();
    return p;
}
Matrix Matrix :: operator - (const Matrix &M)
{
    if (rows != M.rows && cols != M.cols)
    {
        printf ("Matrix1 (%dX%d); Matrix2 (%dX%d)\n", rows, cols, M.rows, M.cols);
        return Matrix ();
    }
    Matrix p (rows, M.cols);
    dim3 block (1, 1, 1);
    dim3 grid (rows, M.cols, 1);
    sub_GPU <<< block, grid>>> (device_pointer, M.device_pointer, p.device_pointer, rows, cols);
    cudaDeviceSynchronize ();
    p.D2H ();
    // p.display ();
    return p;
}
Matrix Matrix :: operator * (const Matrix &M)
{
    if (cols != M.rows)
    {
        printf ("Matrix1 (%dX%d); Matrix2 (%dX%d)\n", rows, cols, M.rows, M.cols);
        return Matrix ();
    }
    Matrix p (rows, M.cols);
    dim3 block (1, 1, 1);
    dim3 grid (rows, M.cols, 1);
    mul_GPU <<< block, grid>>> (device_pointer, M.device_pointer, p.device_pointer, rows, cols, M.cols);
    cudaDeviceSynchronize ();
    p.D2H ();
    // p.display ();
    return p;
}
Matrix Matrix :: operator ~ ()
{
    Matrix t (cols, rows);
    dim3 block (1, 1, 1);
    dim3 grid (rows, cols, 1);
    trp_GPU <<<grid, block>>> (device_pointer, t.device_pointer, rows, cols);
    cudaDeviceSynchronize ();
    t.D2H ();
    return t;
}

__global__ void init_GPU (double *p, int rows, int cols)
{
    int r = threadIdx.x + blockIdx.x * blockDim.x; // x = rows
    int c = threadIdx.y + blockIdx.y * blockDim.y; // y = cols
    // printf ("%d;%d;%d;%d\n", r, c, M.rows, M.cols);
    if (r < rows && c < cols)
    {
        // printf ("<%d>", r * M.cols + c);
        p[r * cols + c] = ((double) (r * cols + c));
        // printf ("%lf ", M.device_pointer[r * M.cols + c]);
    }
    return;
}
void init (double *p, int rows, int cols)
{
    for (int i = 0; i < rows * cols; i++)
    {
        p[i] = rand () % 21 - 10;
    }
    return;
}
__device__ double add_GPU_dev (double m1, double m2)
{
    return m1 + m2;
}
__global__ void add_GPU (double *m1, double *m2, double *a, int rows, int cols)
{
    int Row = blockIdx.x * blockDim.x + threadIdx.x;
    int Col = blockIdx.y * blockDim.y + threadIdx.y;
    if (Row < rows && Col < cols)
    {
        a[Row * cols + Col] = add_GPU_dev (m1[Row * cols + Col], m2[Row * cols + Col]);
    }
    return;
}
__global__ void sub_GPU (double *m1, double *m2, double *a, int rows, int cols)
{
    int Row = blockIdx.x * blockDim.x + threadIdx.x;
    int Col = blockIdx.y * blockDim.y + threadIdx.y;
    if (Row < rows && Col < cols)
    {
        a[Row * cols + Col] = m1[Row * cols + Col] - m2[Row * cols + Col];
    }
    return;
}
__global__ void mul_GPU (double *m1, double *m2, double *p, int rows, int x, int cols)
{
    int Row = blockIdx.x * blockDim.x + threadIdx.x;
    int Col = blockIdx.y * blockDim.y + threadIdx.y;
    if (Row < rows && Col < cols)
    {
        // printf ("{%d,%d}", Row, Col);
        double a = 0;
        for (int k = 0; k < x; k++)
        {
            // printf ("(%.0f,%.0f)", m1[Row * cols + k], m2[k * rows + Col]);
            a += m1[Row * x + k] * m2[k * cols + Col];
        }
        p[Row * cols + Col] = a;
        // printf ("=<%f>\n", a);
    }
    return;
}
__global__ void trp_GPU (double *m1, double *m2, int rows, int cols)
{
    int Row = blockIdx.x * blockDim.x + threadIdx.x;
    int Col = blockIdx.y * blockDim.y + threadIdx.y;
    if (Row < rows && Col < cols)
    {
        m2[Col * rows + Row] = m1[Row * cols + Col];
    }
    return;
}