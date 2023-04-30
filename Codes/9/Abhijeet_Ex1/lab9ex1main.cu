#include "lab9ex1lib.cuh"

int main()
{
	cudaDeviceProp prop;
	int whichDevice;
	CHECK(cudaGetDevice(&whichDevice));
	CHECK(cudaGetDeviceProperties(&prop, whichDevice));
	if(!prop.deviceOverlap) {
		printf( "Device will not handle overlaps, so no speed up from streams\n");
		return 0;
	}
	
	cudaStream_t stream0, stream1;
	CHECK(cudaStreamCreate(&stream0));
	CHECK(cudaStreamCreate(&stream1));

	rec_init;

	// a is array, b is sum, c is max
	int *host_a, *host_sum, *host_max;
	int *dev_a0, *dev_b0;
	int *dev_a1, *dev_b1;
	
	int bdimx = BD;
	int gdimx = (N + bdimx -1)/bdimx;
	dim3 block(bdimx);
	dim3 grid(gdimx);

	printf("Array Size is = %llu\n",N);
	// #pragma push_macro("CHECK")
	// #undef CHECK
	// #define CHECK(Call) Call
	CHECK(cudaMalloc((void**)&dev_a0, N*sizeof(int)));
	CHECK(cudaMalloc((void**)&dev_b0, gdimx*sizeof(int)));

	CHECK(cudaMalloc((void**)&dev_a1, N*sizeof(int)));
	CHECK(cudaMalloc((void**)&dev_b1, gdimx*sizeof(int)));

	CHECK(cudaHostAlloc((void **)&host_a, N * sizeof(int), cudaHostAllocDefault));
	CHECK(cudaHostAlloc((void **)&host_sum, gdimx * sizeof(int), cudaHostAllocDefault));
	CHECK(cudaHostAlloc((void **)&host_max, gdimx * sizeof(int), cudaHostAllocDefault));
	srand(time(0));
	initialize(host_a);
	show(host_a,N);

/*---------------------------------------------------------------------------------------------------------------------*/

	rec_start(stream0);
		CHECK(cudaMemcpyAsync(dev_a0, host_a, N*sizeof(int),cudaMemcpyHostToDevice, stream0));
		CHECK(cudaStreamSynchronize(stream0));
	rec_stop(stream0);
	rec_pr("Time to do memory transfers from H2D in stream0:");

	rec_start(stream0);
		sumReduce<<<grid,block,0,stream0>>>(dev_a0,dev_b0);
		sumReduce<<<1,block,0,stream0>>>(dev_b0,dev_b0);
	rec_stop(stream0);
	rec_pr("TimeElapsed in computation in stream0:");

	rec_start(stream0);
		CHECK(cudaMemcpyAsync(host_sum, dev_b0, sizeof(int),cudaMemcpyDeviceToHost, stream0));
		CHECK(cudaStreamSynchronize(stream0));
	rec_stop(stream0);
	rec_pr("Time to do memory transfers from D2H in stream0:");

	CHECK(cudaStreamSynchronize(stream0));
	printf("Sum = %d\n",host_sum[0]);

/*---------------------------------------------------------------------------------------------------------------------*/

	rec_start(stream1);
		CHECK(cudaMemcpyAsync(dev_a1, host_a, N*sizeof(int),cudaMemcpyHostToDevice, stream1));
		CHECK(cudaStreamSynchronize(stream1));
	rec_stop(stream1);
	rec_pr("Time to do memory transfers from H2D in stream1:");

	rec_start(stream1);
		arrMax<<<grid,block,0,stream1>>>(dev_a1,dev_b1);
		arrMax<<<1,block,0,stream1>>>(dev_b1,dev_b1);
	rec_stop(stream1);
	rec_pr("TimeElapsed in computation in stream1:");

	rec_start(stream1);
		CHECK(cudaMemcpyAsync(host_max, dev_b1, sizeof(int),cudaMemcpyDeviceToHost, stream1));
		CHECK(cudaStreamSynchronize(stream1));
	rec_stop(stream1);
	rec_pr("Time to do memory transfers from D2H in stream0:");

	
	CHECK(cudaStreamSynchronize(stream1));
	printf("Max = %d\n",host_max[0]);

/*---------------------------------------------------------------------------------------------------------------------*/
	// #pragma pop_macro("CHECK")

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(dev_a0);
	cudaFree(dev_b0);
	cudaFree(dev_a1);
	cudaFree(dev_b1);
	cudaFreeHost(host_a);
	cudaFreeHost(host_sum);
	cudaFreeHost(host_max);
	CHECK(cudaStreamDestroy(stream0));
	CHECK(cudaStreamDestroy(stream1));
	cudaDeviceReset();
	return 0;
}
