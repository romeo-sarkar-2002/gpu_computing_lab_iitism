#ifndef WHATEVER_H_INCLUDED
#define WHATEVER_H_INCLUDED

#include <cuda_runtime.h>
#include <stdio.h>
// for 0 <= N <= 2^16
// typedef int integer;
// #define int long long
#define N 20
#define BD 256
#define CHECK(call)\
{\
	const cudaError_t error = call;\
	if(error != cudaSuccess){\
		fprintf(stderr, "Error: %s:%d, ",__FILE__, __LINE__);\
		fprintf(stderr,"code:%d,reason:%s\n",error,\
		cudaGetErrorString(error));\
		exit(1);\
	}\
}
// #define CHECK(call) call
#define rec_init float elapsedTime;\
	cudaEvent_t start, stop;\
	CHECK(cudaEventCreate(&start));\
	CHECK(cudaEventCreate(&stop))

#define rec_start(st) CHECK(cudaEventRecord(start,st))

#define rec_stop(st) CHECK(cudaEventRecord(stop,st));\
	CHECK(cudaEventSynchronize(stop));\
	cudaEventElapsedTime(&elapsedTime,start,stop)

#define rec_pr(s) printf(s" %3.6f ms\n",elapsedTime)

// #define float int
__global__ void sumReduce(int *dev_a,int *dev_sum);
__global__ void arrMax(int *dev_a,int *dev_max);
void show(int *a, int size);
void initialize(int *a);
// #undef float

#endif