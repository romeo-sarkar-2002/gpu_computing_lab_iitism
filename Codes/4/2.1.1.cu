#include <stdio.h>
#include <cuda_runtime.h>

#define precisionField 0

__global__ void mul_GPU (double *m1, double *m2, double *p, int rows, int x, int cols);
void init (double *p, int rows, int cols);

struct Matrix
{
    int rows, cols;
    double *device_pointer, *host_pointer;
    Matrix () : rows (0), cols (0), device_pointer (NULL), host_pointer (NULL)
    {
        return;
    }
    Matrix (int r, int c) : Matrix ()
    {
        rows = r;
        cols = c;
        alloc ();
        return;
    }
    Matrix (const Matrix &M)
    {
        rows = M.rows;
        cols = M.cols;
        cudaMalloc (&device_pointer, rows * cols * sizeof (double));
        cudaMemcpy (device_pointer, M.device_pointer, rows * cols * sizeof (double), cudaMemcpyDeviceToDevice);
        host_pointer = (double *) (malloc (rows * cols * sizeof (double)));
        memcpy (host_pointer, M.host_pointer, rows * cols * sizeof (double));
        return;
    }
    Matrix (Matrix &&M)
    {
        rows = M.rows;
        cols = M.cols;
        device_pointer = M.device_pointer;
        host_pointer = M.host_pointer;
        M.rows = M.cols = 0;
        M.device_pointer = M.host_pointer = NULL;
        return;
    }
    Matrix operator = (Matrix &M)
    {
        clear ();
        rows = M.rows;
        cols = M.cols;
        cudaMalloc (&device_pointer, rows * cols * sizeof (double));
        cudaMemcpy (device_pointer, M.device_pointer, rows * cols * sizeof (double), cudaMemcpyDeviceToDevice);
        host_pointer = (double *) (malloc (rows * cols * sizeof (double)));
        memcpy (host_pointer, M.host_pointer, rows * cols * sizeof (double));
        return *this;
    }
    Matrix operator = (Matrix &&M)
    {
        rows = M.rows;
        cols = M.cols;
        device_pointer = M.device_pointer;
        host_pointer = M.host_pointer;
        M.rows = M.cols = 0;
        M.device_pointer = M.host_pointer = NULL;
        return *this;
    }
    ~Matrix ()
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
    void alloc ()
    {
        cudaMalloc (&device_pointer, rows * cols * sizeof (double));
        host_pointer = (double *) (malloc (rows * cols * sizeof (double)));
        return;
    }
    void clear ()
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
    void display ()
    {
        int *max_width_arr = (int *) (malloc (cols * sizeof (int)));
        char **mat_of_strs = (char **) malloc (rows * cols * sizeof (char *));
        char *str;
        int width;
        for (size_t i = 0; i < cols; i++)
        {
            max_width_arr[i] = 1;
            for (size_t j = 0; j < rows; j++)
            {
                str = (char *) malloc (128 * sizeof (char));
                width = snprintf (str, 128, "%.*lf", precisionField, host_pointer[j * cols + i]);
                str = (char *) realloc (str, ((size_t) (width + 1)) * sizeof (char));
                mat_of_strs[j * cols + i] = str;
                if (max_width_arr[i] < width)
                    max_width_arr[i] = width;
            }
        }
        for (size_t i = 0; i < rows; i++)
        {
            printf ("\xb3");
            for (size_t j = 0; j < cols; j++)
            {
                width = strlen (mat_of_strs[i * cols + j]);
                for (int x = 0; x < max_width_arr[j] - width; x++)
                    printf (" ");
                printf ("%s", mat_of_strs[i * cols + j]);
                if (j != (cols - 1))
                    printf (" ");
            }
            printf ("\xb3");
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
    void init ()
    {
        ::init (host_pointer, rows, cols);
        H2D ();
        return;
    }
    void H2D () // Transfer Data from host to device
    {
        cudaMemcpy (device_pointer, host_pointer, cols * rows * sizeof (double), cudaMemcpyHostToDevice);
        return;
    }
    void D2H () // Transfer Data from device to host
    {
        cudaMemcpy (host_pointer, device_pointer, cols * rows * sizeof (double), cudaMemcpyDeviceToHost);
        return;
    }
    Matrix operator * (const Matrix &M)
    {
        if (cols != M.rows)
        {
            printf ("Matrix1 (%dX%d); Matrix2 (%dX%d)\n", rows, cols, M.rows, M.cols);
            return Matrix ();
        }
        Matrix p (rows, M.cols);
        dim3 block (1, 1, 1);
        dim3 grid (rows, M.cols, 1);
        mul_GPU <<<grid, block>>> (device_pointer, M.device_pointer, p.device_pointer, rows, cols, M.cols);
        cudaDeviceSynchronize ();
        p.D2H ();
        return p;
    }
};
void init (double *p, int rows, int cols)
{
    for (int i = 0; i < rows * cols; i++)
    {
        p[i] = rand () % 21 - 10;
    }
    return;
}
__global__ void mul_GPU (double *m1, double *m2, double *p, int rows, int x, int cols)
{
    int Row = blockIdx.x * blockDim.x + threadIdx.x;
    int Col = blockIdx.y * blockDim.y + threadIdx.y;
    if (Row < rows && Col < cols)
    {
        double a = 0;
        for (int k = 0; k < x; k++)
        {
            a += m1[Row * x + k] * m2[k * cols + Col];
        }
        p[Row * cols + Col] = a;
    }
    return;
}
int main ()
{
    Matrix A (4, 4), B (4, 4), C (4, 4);
    A.init (), B.init (), C.init ();
    printf ("Matrix A:\n");
    A.display ();
    printf ("Matrix B:\n");
    B.display ();
    printf ("Matrix C:\n");
    C.display ();
    Matrix D = A * B * C;
    printf ("Matrix D (A * B * C):\n");
    D.display ();
    
    cudaDeviceReset ();
    return 0;

}