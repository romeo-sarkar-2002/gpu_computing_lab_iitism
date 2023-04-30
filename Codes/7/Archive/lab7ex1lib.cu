#include "lab7ex1lib.cuh"

__device__ void sumReduce(float *dev_a,float *dev_sum) // with warp divergence
{
	__shared__ float partialSum[BD];
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

__device__ void sumReduceLWD(float *dev_a,float *dev_sum) // with less warp divergence
{
	__shared__ float partialSum[BD];
	partialSum[threadIdx.x] = dev_a[blockIdx.x*blockDim.x + threadIdx.x];
	unsigned int t = threadIdx.x;

	for(unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
	{
		__syncthreads();
		if(t < stride)
		{
			partialSum[t] += partialSum[t+stride];
		}
	}
	if(t == 0) dev_sum[blockIdx.x] = partialSum[0];
}


void initialize(float *a){
	for (int i = 0; i < N; i++)
	{
		a[i] = i + 1;
	}
}