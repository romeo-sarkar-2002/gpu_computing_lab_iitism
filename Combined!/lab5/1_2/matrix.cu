#include <stdio.h>
#include <cuda_runtime.h>
#include "matrix.cuh"

#define PRECISION_FIELD 1
#define SHOW_FUNCTION_CALLS 0

__global__ void mulKernel (double *m1, double *m2, double *p, int _rows, int _x, int _cols);
__global__ void transposeKernel (double *m1, double *m2, int _rows, int _cols);


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
    clear ();
    return;
}
void Matrix :: alloc ()
{
    host_pointer = (double *) (malloc (rows * cols * sizeof (double)));
    cudaMalloc (&device_pointer, rows * cols * sizeof (double));
    return;
}
void Matrix :: clear ()
{
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
void Matrix :: display ()
{
    if (NULL == host_pointer)
    {
        return;
    }
    #define BUFFER_SIZE 128
    int *max_width_arr = (int *) (malloc (cols * sizeof (int)));
    char **mat_of_strs = (char **) malloc (rows * cols * sizeof (char *));
    char *str;
    int width;
    for (size_t i = 0; i < cols; i++)
    {
        max_width_arr[i] = 1;
        for (size_t j = 0; j < rows; j++)
        {
            str = (char *) malloc (BUFFER_SIZE * sizeof (char));
            width = snprintf (str, BUFFER_SIZE, "%.*lf", PRECISION_FIELD, host_pointer[j * cols + i]);
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
void Matrix :: initialize ()
{
    ::initialize (host_pointer, rows, cols);
    H2D ();
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
    mulKernel <<< block, grid>>> (device_pointer, M.device_pointer, p.device_pointer, rows, cols, M.cols);
    cudaDeviceSynchronize ();
    p.D2H ();
    return p;
}
Matrix Matrix :: operator ~ ()
{
    Matrix t (cols, rows);
    dim3 block (1, 1, 1);
    dim3 grid (rows, cols, 1);
    transposeKernel <<<grid, block>>> (device_pointer, t.device_pointer, rows, cols);
    cudaDeviceSynchronize ();
    t.D2H ();
    return t;
}
void initialize (double *p, int rows, int cols)
{
    for (int i = 0; i < rows * cols; i++)
    {
        p[i] = rand () % 21 - 10;
    }
    return;
}

__global__ void mulKernel (double *m1, double *m2, double *p, int _rows, int _x, int _cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < _rows && col < _cols)
    {
        double a = 0;
        for (int k = 0; k < _x; k++)
        {
            a += m1[row * _x + k] * m2[k * _cols + col];
        }
        p[row * _cols + col] = a;
    }
    return;
}
__global__ void transposeKernel (double *m1, double *m2, int _rows, int _cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < _rows && col < _cols)
    {
        m2[col * _rows + row] = m1[row * _cols + col];
    }
    return;
}

