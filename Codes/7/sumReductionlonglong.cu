#include <stdio.h>
// macros
#define ceil_div(a, b) ((a + b - 1) / b)
#define floor_div(a, b) (a / b)
#define BLOCK_SIZE 543U

#define chkError(param) \
{ \
    cudaError_t err = (param); \
    if (err != cudaSuccess) \
    { \
        printf ("%s(\033[1;32m%d\033[m): \033[1;4;31merror\033[m: \033[1;33m%s\033[m i.e. %s\n", __FILE__, __LINE__, cudaGetErrorName (err), cudaGetErrorString (err)); \
        exit (err); \
    } \
}
#define getLastError() \
{ \
    cudaError_t err = cudaGetLastError (); \
    if (err != cudaSuccess) \
    { \
        printf ("%s(\033[1;32m%d\033[m): \033[1;4;31merror\033[m: \033[1;33m%s\033[m i.e. %s\n", __FILE__, __LINE__, cudaGetErrorName (err), cudaGetErrorString (err)); \
        exit (err); \
    } \
}
__global__ void reduced_sum (long long *arr, long long *sum, unsigned int size)
{
    __shared__ long long s_arr[BLOCK_SIZE];
    // #define s_arr arr

    unsigned int globalIdx = threadIdx.x + blockIdx.x * BLOCK_SIZE;
    if (globalIdx < size)
    {
        s_arr[threadIdx.x] = arr[globalIdx];
        // if (0 == blockIdx.x)
        // printf ("%.0lf ", arr[globalIdx]);
    }
    __syncthreads ();
    // __syncthreads ();
    // if (threadIdx.x == 0 && blockIdx.x == 0)
    //     printf ("\n\n\n\n");
    // adding the entire block
    unsigned int trailing_stride, stride;
    if (floor_div (size, BLOCK_SIZE) == blockIdx.x)
    {
        // printf ("\033[90mH\033[m");
        trailing_stride = size % BLOCK_SIZE;
    }
    else
    {
        // printf ("\033[90mX\033[m");
        trailing_stride = BLOCK_SIZE;
    }
    stride = ceil_div (trailing_stride, 2);
    for (; trailing_stride > 1; trailing_stride = stride, stride = ceil_div (stride, 2))
    {
        if (threadIdx.x < stride)
        {
            if ((threadIdx.x + stride) < trailing_stride)
            {
                s_arr[threadIdx.x] += s_arr[threadIdx.x + stride];
                // if (blockIdx.x == 0)
                // {
                //     printf ("\033[32m%d->%.0lf\033[m ", threadIdx.x, s_arr[threadIdx.x]);
                // }
            }
            // __syncthreads ();
        }
        else
        {
            goto finish_line;
        }
        // __syncthreads ();
        // if (blockIdx.x == 0 && threadIdx.x == 0)
        // {
        //     printf ("\n\n\n\n");
        // }
        __syncthreads ();
    }
    // if (0 == threadIdx.x)
    // {
    //     printf ("\033[32m%.1lf\033[m ", s_arr[0]);
    // }
    // if (0 == blockIdx.x && threadIdx.x == 0)
    // {
    //     printf ("\033[31m%lf\033[m\n", s_arr[0]);
    // }
    if (0 == threadIdx.x)
    {
        sum[blockIdx.x] = s_arr[0];
    }
    // sum[blockIdx.x] = arr[blockIdx.x * BLOCK_SIZE];
    finish_line:
    return;
}
long long calculate_sum_cpu (long long *arr, size_t size)
{
    long long s = 0;
    for (size_t i = 0; i < size; i++)
    {
        s += arr[i];
    }
    return s;
}
long long calculate_sum_cpu (long long *arr, size_t startIdx, size_t endIdx)
{
    long long s = 0;
    for (int i = startIdx; i < endIdx; i++)
    {
        s += arr[i];
    }
    return s;
}
void initialize_array (long long *arr, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        arr[i] = ((long long) rand ()) * ((long long) (rand ()));
        // arr[i] = i + 1;
    }
    return;
}
int cmp (const void *a, const void *b)
{
    const long long *x = (const long long *) (a), *y = (const long long *) (b);
    if (x < y)
    {
        return 0; // i.e. don't swap
    }
    else
    {
        return 1; // i.e. swap
    }
}
void sort_array (long long *arr, size_t size)
{
    qsort (arr, size, sizeof (long long), cmp);
    return;
}
void print_array (long long *arr, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        printf ("%lld ", arr[i]);
    }
    printf ("\n");
    return;
}

long long calculate_sum_gpu (long long *arr, size_t size)
{
    // long long *dev_arr;
    // cudaMalloc (&dev_arr, size);
    
    // cudaMemcpy (dev_arr, arr, size * sizeof (long long), cudaMemcpyHostToDevice);
    // array will be divided into smaller array of size BLOCK_SIZE
    long long sum;
    size_t temp_arr_size = size, temp_sum_arr_size = ceil_div (size, BLOCK_SIZE);
    long long *dev_temp_arr = NULL, *dev_temp_sum_arr = NULL;
    cudaMalloc (&dev_temp_arr, sizeof (long long) * temp_arr_size);
    cudaMemcpy (dev_temp_arr, arr, temp_arr_size * sizeof (long long), cudaMemcpyHostToDevice);
    for (; temp_arr_size > 1; temp_arr_size = temp_sum_arr_size, temp_sum_arr_size = ceil_div (temp_sum_arr_size, BLOCK_SIZE))
    {
        // temp_size = ceildiv (temp_size, BLOCK_SIZE);
        // temp_arr = (long long *) (malloc (sizeof (long long) * temp_arr_size));
        cudaMalloc (&dev_temp_sum_arr, sizeof (long long) * temp_sum_arr_size);
        reduced_sum <<< temp_sum_arr_size, BLOCK_SIZE >>> (dev_temp_arr, dev_temp_sum_arr, temp_arr_size);
        getLastError ();
        cudaDeviceSynchronize ();
        // printf ("\n");
        // comment:
        // long long *p = (long long *) malloc (temp_sum_arr_size * sizeof (long long));
        // cudaMemcpy (p, dev_temp_sum_arr, sizeof (long long) * temp_sum_arr_size, cudaMemcpyDeviceToHost);
        // long long t;
        // for (int i = 0; i < temp_sum_arr_size && i < 1; i++)
        // {
        //     if (p[i] != (t = calculate_sum_cpu (arr, i * BLOCK_SIZE, (i + 1) * BLOCK_SIZE)))
        //     {
        //         printf ("error: %d; %.1lf instead of %.1lf\n", i, p[i], t);
        //     }
        // }
        // comment:

        cudaFree (dev_temp_arr);
        dev_temp_arr = dev_temp_sum_arr;
        dev_temp_sum_arr = NULL;
        // return 0;
        
    }
    cudaMemcpy (&sum, dev_temp_arr, sizeof (long long), cudaMemcpyDeviceToHost);
    cudaFree (dev_temp_arr);
    return sum;
}
int main ()
{
    srand (time (NULL));
    size_t size = 56578645; // array size;
    long long *arr = (long long *) (malloc (sizeof (long long) * size));
    initialize_array (arr, size);
    sort_array (arr, size);
    printf ("sorted!\n");
    // print_array (arr, size);
    long long sum_cpu, sum_gpu;
    if ((sum_cpu = calculate_sum_cpu (arr, size)) != (sum_gpu = calculate_sum_gpu (arr, size)))
    {
        printf ("\033[1;31merror\033[m: (sum_cpu != sum_gpu)\n");
    }
    printf ("%lld|%lld\n", sum_cpu, sum_gpu);
    printf ("done\n");
    return 0;
}