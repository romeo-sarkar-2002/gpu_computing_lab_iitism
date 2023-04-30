#include "implementation.h"
#include "cuda_runtime.h"
#include "stdio.h"

union float2UllUnion {
	float2 f;
	unsigned long long int ull;
};

__global__ void atomicTests(int *var1, int *var2);
__global__ void bitonicSortStep(float*, int, int);
__global__ void sumKernel(float*, float2*, size_t);

// Implement Bitonic sort
// Only works for powers of 2 for now
void gpu_sort(float *arr, int n) {
	float *dev_a;
	size_t arr_size = n * sizeof(float);

  dim3 blocks(256);
  dim3 grid((blocks.x + n - 1) / blocks.x);

	CHECK(cudaMalloc((void**)&dev_a, arr_size));
	CHECK(cudaMemcpy(dev_a, arr, arr_size, cudaMemcpyHostToDevice));

  for (int k = 2; k <= n; k <<= 1) {
    for (int j = k >> 1; j > 0; j = j >> 1) {
      bitonicSortStep<<<grid, blocks>>>(dev_a, j, k);
    }
  }

	CHECK(cudaMemcpy(arr, dev_a, arr_size, cudaMemcpyDeviceToHost));

	CHECK(cudaFree(dev_a));
  CHECK(cudaDeviceReset());
	return;
}

struct Result kahan_sum(float *arr, size_t n) {
  struct Result ans;
  float *dev_arr;
  float2 *dev_khn;
  float2 host_khn = {0, 0};
	size_t arr_size = n * sizeof(float);

  dim3 blocks(256);
  dim3 grid((blocks.x + n - 1) / blocks.x);

  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));

  CHECK(cudaEventRecord(start, 0));

  CHECK(cudaMalloc((void**)&dev_arr, arr_size));
  CHECK(cudaMalloc((void**)&dev_khn, sizeof(float2)));
  
	CHECK(cudaMemcpy(dev_arr, arr, arr_size, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(dev_khn, &host_khn, sizeof(float2), cudaMemcpyHostToDevice));

  sumKernel<<<grid,  blocks>>>(dev_arr, dev_khn, n);

  CHECK(cudaMemcpy(&host_khn, dev_khn, sizeof(float2), cudaMemcpyDeviceToHost));

  CHECK(cudaFree(dev_arr));
  CHECK(cudaFree(dev_khn));

  CHECK(cudaEventRecord(stop, 0));
  CHECK(cudaEventSynchronize(stop));
  CHECK(cudaEventElapsedTime(&ans.time, start, stop));
  CHECK(cudaDeviceReset());

  ans.ans = host_khn.x;

  return ans;
}

float khn_sum_host(float *arr, int n) {
  float sum = 0;
  float c = 0;
  for(int i = 0; i < n; i++) {
    float y = arr[i] - c;
    float t = sum + y;
    c = t - sum - y;
    sum = t;
  }
  return sum;
}

void atomic_tests(int *var1, int *var2) {
	int *dev_var1;
	int *dev_var2;

	CHECK(cudaMalloc((void **)&dev_var1, sizeof(int)));
	CHECK(cudaMalloc((void **)&dev_var2, sizeof(int)));
	CHECK(cudaMemcpy(dev_var1, var1, sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(dev_var2, var2, sizeof(int), cudaMemcpyHostToDevice));

	atomicTests<<<1, 2>>>(dev_var1, dev_var2);

	CHECK(cudaMemcpy(var1, dev_var1, sizeof(int), cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(var2, dev_var2, sizeof(int), cudaMemcpyDeviceToHost));

	CHECK(cudaDeviceSynchronize());

	CHECK(cudaFree(dev_var1));
	CHECK(cudaFree(dev_var2));
  CHECK(cudaDeviceReset());

	return;
}

__global__ void atomicTests(int *var1, int *var2) {
	atomicAdd(var1, 1);
	atomicSub(var2, 1);
}

__device__ void swap(float *a, float *b) {
  float temp = *a;
  *a = *b;
  *b = temp;
}

__global__ void bitonicSortStep(float *dev_values, int j, int k)
{
  unsigned int index1 = threadIdx.x + blockDim.x * blockIdx.x;
  unsigned int index2 = index1 ^ j;

  /* The threads with the lowest ids sort the array. */
  if (index2 > index1) {
    if ((index1 & k) == 0) {
      /* Sort ascending */
      if (dev_values[index1] > dev_values[index2]) {
        /* exchange(i,ixj); */
        swap(&dev_values[index1], &dev_values[index2]);
      }
    }
    if ((index1 & k)!=0) {
      /* Sort descending */
      if (dev_values[index1] < dev_values[index2]) {
        /* exchange(i,ixj); */
        swap(&dev_values[index1], &dev_values[index2]);
      }
    }
  }
}



__device__ void atomicKhn(const float val, float2* __restrict address)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    float2UllUnion old, assumed, tmp;
    old.ull = *address_as_ull;
    do {
        assumed = old;
        tmp = assumed;

        const float y = val - tmp.f.y;
        const float t = tmp.f.x + y;
        tmp.f.y = (t - tmp.f.x) - y;
        tmp.f.x = t;

        old.ull = atomicCAS(address_as_ull, assumed.ull, tmp.ull);
    } while (assumed.ull != old.ull);

}

__global__ void sumKernel(float *arr, float2 *khn, size_t len) { 
  size_t index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < len) {
    atomicKhn(arr[index], khn);
  }
}
