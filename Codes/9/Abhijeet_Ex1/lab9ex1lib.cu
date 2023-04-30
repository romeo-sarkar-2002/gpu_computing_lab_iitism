#include "lab9ex1lib.cuh"

__global__ void sumReduce(int *dev_a,int *dev_sum) // with warp divergence
{
	__shared__ int partialSum[BD];
	partialSum[threadIdx.x] = dev_a[blockIdx.x*blockDim.x + threadIdx.x];
	unsigned int t = threadIdx.x;

	for(unsigned int stride = 1; stride < blockDim.x; stride <<= 1)
	{
		__syncthreads();
		if(t % (2*stride) == 0)
		{
			partialSum[t] += partialSum[t+stride];
		}
	}
	if(t == 0) dev_sum[blockIdx.x] = partialSum[0];
}
__global__ void arrMax(int *dev_a,int *dev_max){
	__shared__ int partialMax[BD];
	partialMax[threadIdx.x] = dev_a[blockIdx.x*blockDim.x + threadIdx.x];
	unsigned int t = threadIdx.x;

	for(unsigned int stride = 1; stride < blockDim.x; stride <<= 1)
	{
		__syncthreads();
		if(t % (2*stride) == 0)
		{
			// partialMax[t] = (partialMax[t] > partialMax[t+stride])? partialMax[t]:partialMax[t+stride];
			partialMax[t] = max(partialMax[t], partialMax[t+stride]);
		}
	}
	if(t == 0) dev_max[blockIdx.x] = partialMax[0];
}

void initialize(int *a){
	for (int i = 0; i < N; i++)
	{
		// a[i] = i + 1;
		a[i] = rand() % 100;
	}
}

void show(int *a, int size){
	for (int i = 0; i < size; i++)
	{
		printf("%d ",a[i]);
	}
	printf("\n");
}