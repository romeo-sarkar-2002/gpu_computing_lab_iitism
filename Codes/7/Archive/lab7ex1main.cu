#include <cuda_runtime.h>
#include <stdio.h>
#include "lab7ex1lib.cu"
#include "lab7ex1lib.cuh"

__global__ void sr(int option, float *dev_a,float *dev_sum){
	// 0 for some reduction with warp divergence
	// 1 for some reduction with less warp divergence
	if(option){
		sumReduceLWD(dev_a, dev_sum);
	}
	else{
		sumReduce(dev_a, dev_sum);
	}
}

int main()
{
	// variable declarition
	int bdimx = BD;
	int gdimx = (N + bdimx -1)/bdimx;
	dim3 block(bdimx);
	dim3 grid(gdimx);

	// declaring input array(a) and sum array(b) for both host and device
	float *a,*b,*c;
	float *dev_a,*dev_b,*dev_c;
	
	a = (float *)malloc(N*sizeof(float));
	b = (float *)malloc(gdimx*sizeof(float));
	c = (float *)malloc(gdimx*sizeof(float));
	// setup for measure time elapsed
	rec_init;

	// allocate the memory on device
	CHECK(cudaMalloc((void**)&dev_a, N*sizeof(float)));
	CHECK(cudaMalloc((void**)&dev_b, gdimx*sizeof(float)));
	CHECK(cudaMalloc((void**)&dev_c, gdimx*sizeof(float)));

	// initilize array a
	initialize(a);

	rec_start;
		// copying array data to device
		CHECK(cudaMemcpy(dev_a, a, N*sizeof(float),cudaMemcpyHostToDevice));
	rec_stop;

	// printing array size and time elapsed for memory transfer
	printf("Array Size is = %d\n",N);
	rec_pr("Time to do memory transfer of array a, from host to device is");
	
	rec_start;
		//kernel launch for some reduction with warp divergence
		sr<<<grid,block>>>(0,dev_a,dev_b);
		sr<<<1,block>>>(0,dev_b,dev_b);
		cudaDeviceSynchronize();
		CHECK(cudaMemcpy(b,dev_b, sizeof(float),cudaMemcpyDeviceToHost));
	rec_stop;
	rec_pr("Time to do sum reduction with warp divergence is");
	printf("Sum = %f\n",b[0]);

	rec_start;
		//kernel launch for some reduction with less warp divergence
		sr<<<grid,block>>>(1,dev_a,dev_c);
		sr<<<1,block>>>(1,dev_c,dev_c);
		cudaDeviceSynchronize();
		CHECK(cudaMemcpy(c,dev_c, sizeof(float),cudaMemcpyDeviceToHost));
	rec_stop;
	rec_pr("Time to do sum reduction with less warp divergence is");
	printf("Sum = %f\n",c[0]);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(dev_a);
	cudaFree(dev_b);
	return 0;
}
