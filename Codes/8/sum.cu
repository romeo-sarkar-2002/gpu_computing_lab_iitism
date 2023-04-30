#include <stdio.h>
#include "error.cuh"
#define _SHARED_ARR_LEN_ 347
#define ceil_div(a, b) (((a) + (b) - 1) / (b))
#define floor_div(a, b) ((a) / (b))
__global__ void reduced_sum (double *arr, double *sum, size_t size)
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
int cmp (const void *a, const void *b)
{
    const double *x = (const double *) (a), *y = (const double *) (b);
    if ((*x) < (*y))
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
    for (int i = 0; i < 10; i++)
    {
        printf ("%.9lf ", arr[i]);
    }
    printf ("\n");
}
double calculate_sum_gpu (double *arr, size_t size)
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
        reduced_sum <<< temp_sum_arr_size, _SHARED_ARR_LEN_ >>> (dev_temp_arr, dev_temp_sum_arr, temp_arr_size);
        getLastError ();
        cudaDeviceSynchronize ();

        cudaFree (dev_temp_arr);
        dev_temp_arr = dev_temp_sum_arr;
        dev_temp_sum_arr = NULL;
        // return 0;
        
    }
    cudaMemcpy (&sum, dev_temp_arr, sizeof (double), cudaMemcpyDeviceToHost);
    cudaFree (dev_temp_arr);
    return sum;
}
double *allocate_array (size_t size)
{
    printf ("size of array: %zd Bytes (%.6lf GB)\n", sizeof (double) * size, ((double) (sizeof (double) * size)) / (1024.0 * 1024.0 * 1024.0));
    double *arr = (double *) (malloc (sizeof (double) * size));
    return arr;
}
void initialize_array (double *arr, size_t size)
{
    struct timespec start, stop;
    timespec_get (&start, TIME_UTC);
    for (size_t i = 0; i < size; i++)
    {
        arr[i] = ((double) rand ()) * ((double) (rand ()));
        // if(i&1) arr[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/1024));
		// else arr[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX));
        // arr[i] = ((double) rand ());
        // arr[i] = i + 1;
        // arr[i] = 1;
    }
    timespec_get (&stop, TIME_UTC);
    printf ("time taken to initialize the array: %.9lf secs.\n", ((double) (stop.tv_nsec - start.tv_nsec) * 1e-9 + ((double) (stop.tv_sec - start.tv_sec))));
    return;
}
int sum (double *arr, size_t size)
{
    struct timespec start, stop;
    print_array (arr, size);
    timespec_get (&start, TIME_UTC);
    double gpuSum = calculate_sum_gpu (arr, size);
    // double gpuSum = calculate_sum_gpu ()
    timespec_get (&stop, TIME_UTC);

    // clock_t st = clock ();
    sort_array (arr, size);
    print_array (arr, size);
    // printf ("time: %.3lf secs.\n", ((double) (clock () - st)) / CLOCKS_PER_SEC);
    // printf ("time taken to sort the array: %.9lf secs.\n", ((double) (stop.tv_nsec - start.tv_nsec)) * 1e-9 + ((double) (stop.tv_sec - start.tv_sec)));
    /* = = = = = = = = = = = = = */
    timespec_get (&start, TIME_UTC);
    double gpuSumSorted = calculate_sum_gpu (arr, size);
    timespec_get (&stop, TIME_UTC);
    // printf ("sum_cpu time: %.9lf secs.\n", ((double) (stop.tv_nsec - start.tv_nsec) * 1e-9 + ((double) (stop.tv_sec - start.tv_sec))));
    /*= = = = = = = = = = = = = */
    // timespec_get (&start, TIME_UTC);
    // timespec_get (&stop, TIME_UTC);
    // printf ("sum_gpu1 time: %.9lf secs.\n", ((double) (stop.tv_nsec - start.tv_nsec) * 1e-9 + ((double) (stop.tv_sec - start.tv_sec))));
    /*= = = = = = = = = = = = = */
    printf ("{gpuSum, gpuSumSorted} = {%.0lf, %.0lf}\n", gpuSum, gpuSumSorted);
    return 0;
}
int main ()
{
    srand (time (NULL));
    size_t size = 457634789; // array size;
    double *arr = allocate_array (size);
    initialize_array (arr, size);
    sum (arr, size);
    return 0;
}