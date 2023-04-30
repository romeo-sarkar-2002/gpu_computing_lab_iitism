#include <stdio.h>
// macros
#define ceil_div(a, b) ((a + b - 1) / b)
#define floor_div(a, b) (a / b)
#define BLOCK_SIZE 1023U

__global__ void reduced_sum (double *arr, double *sum, unsigned int size)
{
    // __shared__ double s_arr[BLOCK_SIZE];
    #define s_arr arr

    unsigned int globalIdx = threadIdx.x + blockIdx.x * BLOCK_SIZE;
    // if (globalIdx < size)
    // {
    //     s_arr[threadIdx.x] = arr[globalIdx];
    // }
    // adding the entire block
    unsigned int trailing_stride, stride;
    if (floor_div (size, BLOCK_SIZE) == blockIdx.x)
    {
        trailing_stride = size % BLOCK_SIZE;
    }
    else
    {
        trailing_stride = BLOCK_SIZE;
    }
    stride = ceil_div (trailing_stride, 2);
    for (; trailing_stride > 1; trailing_stride = stride, stride = ceil_div (stride, 2))
    {
        if (threadIdx.x < stride)
        {
            if (threadIdx.x + stride < trailing_stride)
            {
                s_arr[globalIdx] += s_arr[globalIdx + stride];
            }
        }
        else
        {
            goto finish_line;
        }
        __syncthreads ();
    }
    // if (0 == threadIdx.x)
    // {
    //     printf ("\033[32m%.1lf\033[m\n", s_arr[0]);
    // }
    // sum[blockIdx.x] = s_arr[0];
    sum[blockIdx.x] = arr[blockIdx.x * BLOCK_SIZE];
    finish_line:
    return;
}
double calculate_sum_cpu (double *arr, size_t size)
{
    double s = 0;
    for (size_t i = 0; i < size; i++)
    {
        s += arr[i];
    }
    return s;
}
void initialize (double *arr, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        // arr[i] = rand ();
        arr[i] = i + 1;
    }
    return;
}
void print (double *arr, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        printf ("%.1f ", arr[i]);
    }
    printf ("\n");
    return;
}
double calculate_sum_gpu (double *arr, size_t size)
{
    // double *dev_arr;
    // cudaMalloc (&dev_arr, size);
    
    // cudaMemcpy (dev_arr, arr, size * sizeof (double), cudaMemcpyHostToDevice);
    // array will be divided into smaller array of size BLOCK_SIZE
    printf ("\033[31m%zd\033[m\n", ceil_div (size, BLOCK_SIZE));

    size_t temp_arr_size = size, temp_sum_arr_size = ceil_div (size, BLOCK_SIZE);
    double *dev_temp_arr = NULL, *dev_temp_sum_arr = NULL;
    cudaMalloc (&dev_temp_arr, sizeof (double) * temp_arr_size);
    cudaMalloc (&dev_temp_sum_arr, temp_sum_arr_size * sizeof (double));
    cudaMemcpy (dev_temp_arr, arr, size * sizeof (double), cudaMemcpyHostToDevice);
    // for (; ; temp_arr_size = temp_sum_arr_size, temp_sum_arr_size = ceildiv (temp_sum_arr_size, BLOCK_SIZE))
    // {
    //     // temp_size = ceildiv (temp_size, BLOCK_SIZE);
    //     // temp_arr = (double *) (malloc (sizeof (double) * temp_arr_size));
    //     cudaMalloc (&dev_temp_sum_arr, sizeof (double) * temp_sum_arr_size);
    //     reduced_sum <<< temp_arr_size, BLOCK_SIZE >>> (dev_temp_arr, dev_temp_sum_arr, temp_arr_size);

    //     cudaFree (dev_temp_arr);
    //     dev_temp_arr = dev_temp_sum_arr;
    // }
    reduced_sum <<< temp_sum_arr_size, BLOCK_SIZE >>> (dev_temp_arr, dev_temp_sum_arr, size);
    double *s = (double *) malloc (sizeof (double) * temp_sum_arr_size);
    cudaMemcpy (s, dev_temp_sum_arr, temp_sum_arr_size * sizeof (double), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < temp_sum_arr_size; i++)
    {
        printf ("%.1lf ", s[i]);
    }
    printf ("\n");
    return 0;
} 
int main ()
{
    srand (time (NULL));
    size_t size = 5243; // array size;
    double *arr = (double *) (malloc (sizeof (double) * size));
    initialize (arr, size);
    double sum;
    // if (calculate_sum_cpu (arr, size) != (sum = calculate_sum_gpu (arr, size)))
    // {
    //     printf ("error: (sum_cpu != sum_gpu)\n");
    // }
    calculate_sum_gpu (arr, size);

    return 0;
}