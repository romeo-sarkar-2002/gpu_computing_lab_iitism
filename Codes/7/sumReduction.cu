#include <stdio.h>
#include <time.h>
// macros
#define _SHARED_ARR_LEN_ 347U

#define ceil_div(a, b) (((a) + (b) - 1) / (b))
#define floor_div(a, b) ((a) / (b))
#include "Error.cuh"

__global__ void reduced_sum1 (double *arr, double *sum, size_t size)
{
    __shared__ double s_arr[_SHARED_ARR_LEN_];
    // unsigned int s = ceil_div (_SHARED_ARR_LEN_, 2);
    // unsigned int globalIdx = 2 * threadIdx.x + blockIdx.x * _SHARED_ARR_LEN_;
    // if (2 * threadIdx.x < )
    unsigned int s;
    if (floor_div (size, 2 * _SHARED_ARR_LEN_) == blockIdx.x)
    {
        s = ceil_div (size % (2 * _SHARED_ARR_LEN_), 2);
    }
    else
    {
        s = _SHARED_ARR_LEN_;
    }
    if ((threadIdx.x < s) && ((2 * (threadIdx.x + blockIdx.x * _SHARED_ARR_LEN_)) < size))
    {
        s_arr[threadIdx.x] = arr[2 * (threadIdx.x + blockIdx.x * _SHARED_ARR_LEN_)];
    }
    // else
    // {
    //     goto finish_line;
    // }
    __syncthreads ();
    if ((threadIdx.x < s) && ((2 * (threadIdx.x + blockIdx.x * _SHARED_ARR_LEN_) + 1) < size))
    {
        s_arr[threadIdx.x] += arr[2 * (threadIdx.x + blockIdx.x * _SHARED_ARR_LEN_) + 1];
    }
    // if (0 == blockIdx.x)
    // {
    //     // for (unsigned int i = 0; i < _SHARED_ARR_LEN_; i++)
    //     // {
    //     printf ("%.0lf ", s_arr[threadIdx.x]);
    //     // }
    //     __syncthreads ();
    //     if (0 == threadIdx.x)
    //         printf ("\n");
    // }
    __syncthreads ();
    
    // unsigned int 
    // now, find the sum of the entire block:
    for (unsigned int stride = 1; stride < s; stride <<= 1)
    {
        if ((threadIdx.x % (stride << 1)) == 0)
        {
            if ((threadIdx.x + stride) < s)
            {
                s_arr[threadIdx.x] += s_arr[threadIdx.x + stride];
            }
        }
        // else
        // {
        //     goto finish_line;
        // }
        __syncthreads ();
    }
    sum[blockIdx.x] = s_arr[0];
    // finish_line:
    // printf ("<<%u;%u>>\n", blockIdx.x, threadIdx.x);
    // blockIdx.x
    return;
}
__global__ void reduced_sum2 (double *arr, double *sum, unsigned int size)
{
    __shared__ double s_arr[_SHARED_ARR_LEN_];
    // #define s_arr arr

    unsigned int globalIdx = threadIdx.x + blockIdx.x * _SHARED_ARR_LEN_;
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
    if (floor_div (size, _SHARED_ARR_LEN_) == blockIdx.x)
    {
        // printf ("\033[90mH\033[m");
        trailing_stride = size % _SHARED_ARR_LEN_;
    }
    else
    {
        // printf ("\033[90mX\033[m");
        trailing_stride = _SHARED_ARR_LEN_;
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
    // sum[blockIdx.x] = arr[blockIdx.x * _SHARED_ARR_LEN_];
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
double calculate_sum_cpu (double *arr, size_t startIdx, size_t endIdx)
{
    double s = 0;
    for (int i = startIdx; i < endIdx; i++)
    {
        s += arr[i];
    }
    return s;
}
void initialize_array (double *arr, size_t size)
{
    struct timespec start, stop;
    timespec_get (&start, TIME_UTC);
    for (size_t i = 0; i < size; i++)
    {
        // arr[i] = ((double) rand ()) * ((double) (rand ()));
        arr[i] = (double) rand ();
        // arr[i] = i + 1;
        // arr[i] = 1;
    }
    timespec_get (&stop, TIME_UTC);
    printf ("time taken to initialize the array: %.9lf secs.\n", ((double) (stop.tv_nsec - start.tv_nsec) * 1e-9 + ((double) (stop.tv_sec - start.tv_sec))));
    return;
}
int cmp (const void *a, const void *b)
{
    const double *x = (const double *) (a), *y = (const double *) (b);
    if (x < y)
    {
        return 0; // i.e. don't swap
    }
    else
    {
        return 1; // i.e. swap
    }
}
void sort_array (double *arr, size_t size)
{
    struct timespec start, stop;
    timespec_get (&start, TIME_UTC);
    qsort (arr, size, sizeof (double), cmp);
    timespec_get (&stop, TIME_UTC);
    printf ("time taken to sort the array: %.9lf secs.\n", ((double) (stop.tv_nsec - start.tv_nsec)) * 1e-9 + ((double) (stop.tv_sec - start.tv_sec)));
    return;
}
void print_array (double *arr, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        printf ("%.0f ", arr[i]);
    }
    printf ("\n");
    return;
}
double calculate_sum_gpu1 (double *arr, size_t size)
{
    // double *dev_arr;
    // cudaMalloc (&dev_arr, size);
    
    // cudaMemcpy (dev_arr, arr, size * sizeof (double), cudaMemcpyHostToDevice);
    // array will be divided into smaller array of size _SHARED_ARR_LEN_
    double sum;
    size_t temp_arr_size = size, temp_sum_arr_size = ceil_div (temp_arr_size, 2 * _SHARED_ARR_LEN_);
    double *dev_temp_arr = NULL, *dev_temp_sum_arr = NULL;
    cudaMalloc (&dev_temp_arr, sizeof (double) * temp_arr_size);
    cudaMemcpy (dev_temp_arr, arr, temp_arr_size * sizeof (double), cudaMemcpyHostToDevice);
    for (; temp_arr_size > 1; temp_arr_size = temp_sum_arr_size, temp_sum_arr_size = ceil_div (temp_arr_size, 2 * _SHARED_ARR_LEN_))
    {
        // temp_size = ceildiv (temp_size, _SHARED_ARR_LEN_);
        // temp_arr = (double *) (malloc (sizeof (double) * temp_arr_size));
        // printf ("launch param: <<< %zu, %u >>>\n", temp_sum_arr_size, _SHARED_ARR_LEN_);
        // printf ("launch param1: %u\n", ceil_div (12, 5));
        chkError (cudaMalloc (&dev_temp_sum_arr, sizeof (double) * temp_sum_arr_size))
        reduced_sum1 <<< temp_sum_arr_size, _SHARED_ARR_LEN_ >>> (dev_temp_arr, dev_temp_sum_arr, temp_arr_size);
        getLastError ();
        cudaDeviceSynchronize ();
        // printf ("launch param2: %zu, %zu\n", temp_arr_size, temp_sum_arr_size);
        // printf ("\n");
        // comment:
        // double *p = (double *) malloc (temp_sum_arr_size * sizeof (double));
        // cudaMemcpy (p, dev_temp_sum_arr, sizeof (double) * temp_sum_arr_size, cudaMemcpyDeviceToHost);
        // double t;
        // for (int i = 0; i < temp_sum_arr_size; i++)
        // {
        //     printf ("\033[31m%.1lf\033[m ", p[i]);
        // }
        // printf ("\n");
        // for (int i = 0; i < temp_sum_arr_size; i++)
        // {
        //     if (p[i] != (t = calculate_sum_cpu (arr, i * 2 * _SHARED_ARR_LEN_, (i + 1) * 2 * _SHARED_ARR_LEN_)))
        //     {
        //         printf ("error: %d; %.1lf instead of %.1lf\n", i, p[i], t);
        //         exit (0);
        //     }
        // }
        // exit (0);
        // comment:

        cudaFree (dev_temp_arr);
        dev_temp_arr = dev_temp_sum_arr;
        dev_temp_sum_arr = NULL;
        // return 0;
        
    }
    cudaMemcpy (&sum, dev_temp_arr, sizeof (double), cudaMemcpyDeviceToHost);
    cudaFree (dev_temp_arr);
    return sum;
}
double calculate_sum_gpu2 (double *arr, size_t size)
{
    // double *dev_arr;
    // cudaMalloc (&dev_arr, size);
    
    // cudaMemcpy (dev_arr, arr, size * sizeof (double), cudaMemcpyHostToDevice);
    // array will be divided into smaller array of size _SHARED_ARR_LEN_
    double sum;
    size_t temp_arr_size = size, temp_sum_arr_size = ceil_div (size, _SHARED_ARR_LEN_);
    double *dev_temp_arr = NULL, *dev_temp_sum_arr = NULL;
    cudaMalloc (&dev_temp_arr, sizeof (double) * temp_arr_size);
    cudaMemcpy (dev_temp_arr, arr, temp_arr_size * sizeof (double), cudaMemcpyHostToDevice);
    for (; temp_arr_size > 1; temp_arr_size = temp_sum_arr_size, temp_sum_arr_size = ceil_div (temp_sum_arr_size, _SHARED_ARR_LEN_))
    {
        // temp_size = ceildiv (temp_size, _SHARED_ARR_LEN_);
        // temp_arr = (double *) (malloc (sizeof (double) * temp_arr_size));
        cudaMalloc (&dev_temp_sum_arr, sizeof (double) * temp_sum_arr_size);
        reduced_sum2 <<< temp_sum_arr_size, _SHARED_ARR_LEN_ >>> (dev_temp_arr, dev_temp_sum_arr, temp_arr_size);
        getLastError ();
        cudaDeviceSynchronize ();
        // printf ("\n");
        // comment:
        // double *p = (double *) malloc (temp_sum_arr_size * sizeof (double));
        // cudaMemcpy (p, dev_temp_sum_arr, sizeof (double) * temp_sum_arr_size, cudaMemcpyDeviceToHost);
        // double t;
        // for (int i = 0; i < temp_sum_arr_size && i < 1; i++)
        // {
        //     if (p[i] != (t = calculate_sum_cpu (arr, i * _SHARED_ARR_LEN_, (i + 1) * _SHARED_ARR_LEN_)))
        //     {
        //         printf ("error: %d; %.1lf instead of %.1lf\n", i, p[i], t);
        //     }
        // }
        // comment:
        /*
        f
        sf
        sd

        */

        cudaFree (dev_temp_arr);
        dev_temp_arr = dev_temp_sum_arr;
        dev_temp_sum_arr = NULL;
        // return 0;
        
    }
    cudaMemcpy (&sum, dev_temp_arr, sizeof (double), cudaMemcpyDeviceToHost);
    cudaFree (dev_temp_arr);
    return sum;
}
int sum (double *arr, size_t size)
{
    struct timespec start, stop;
    // timespec_get (&start, TIME_UTC);
    // clock_t st = clock ();
    sort_array (arr, size);
    // printf ("time: %.3lf secs.\n", ((double) (clock () - st)) / CLOCKS_PER_SEC);
    // timespec_get (&stop, TIME_UTC);
    // printf ("time taken to sort the array: %.9lf secs.\n", ((double) (stop.tv_nsec - start.tv_nsec)) * 1e-9 + ((double) (stop.tv_sec - start.tv_sec)));
    /* = = = = = = = = = = = = = */
    timespec_get (&start, TIME_UTC);
    double sum_cpu = calculate_sum_cpu (arr, size);
    timespec_get (&stop, TIME_UTC);
    printf ("sum_cpu time: %.9lf secs.\n", ((double) (stop.tv_nsec - start.tv_nsec) * 1e-9 + ((double) (stop.tv_sec - start.tv_sec))));
    /*= = = = = = = = = = = = = */
    timespec_get (&start, TIME_UTC);
    double sum_gpu1 = calculate_sum_gpu1 (arr, size);
    timespec_get (&stop, TIME_UTC);
    printf ("sum_gpu1 time: %.9lf secs.\n", ((double) (stop.tv_nsec - start.tv_nsec) * 1e-9 + ((double) (stop.tv_sec - start.tv_sec))));
    /*= = = = = = = = = = = = = */
    timespec_get (&start, TIME_UTC);
    double sum_gpu2 = calculate_sum_gpu2 (arr, size);
    timespec_get (&stop, TIME_UTC);
    printf ("sum_gpu2 time: %.9lf secs. \033[90m(less warp divergence)\033[m\n", ((double) (stop.tv_nsec - start.tv_nsec) * 1e-9 + ((double) (stop.tv_sec - start.tv_sec))));
    /*= = = = = = = = = = = = = */
    printf ("{sum_cpu, sum_gpu1, sum_gpu2} = {%.0lf, %.0lf, %.0lf}\n", sum_cpu, sum_gpu1, sum_gpu2);
    if (sum_cpu != sum_gpu1)
    {
        printf ("\033[1;31merror\033[m: (sum_cpu != sum_gpu1)\n");
        return 1;
    }
    else
    {
        if (sum_cpu != sum_gpu2)
        {
            printf ("\033[1;31merror\033[m: (sum_cpu != sum_gpu2)\n");
            return 1;
        }
        else
        {
            return 0;
        }
    }
    // const int i = 0;
    // if (sum_cpu != sum_gpu2)
    // {
    //     printf ("\033[1;31merror\033[m: (sum_cpu != sum_gpu)\n");
    //     return 1;
    // }
    // else
    // {
    //     return 0;
    // }
    return 0;
}
double *allocate_array (size_t size)
{
    printf ("size of array: %zd Bytes (%.6lf GB)\n", sizeof (double) * size, ((double) (sizeof (double) * size)) / (1024.0 * 1024.0 * 1024.0));
    double *arr = (double *) (malloc (sizeof (double) * size));
    return arr;
}
int main ()
{
    srand (time (NULL));
    size_t size = 565786565; // array size;

    double *arr = allocate_array (size);
    initialize_array (arr, size);
    sum (arr, size);
    return 0;
}