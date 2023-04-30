#ifndef WHATEVER_H_INCLUDED
#define WHATEVER_H_INCLUDED

#define N (1<<16)
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

#define rec_init float elapsedTime;\
	cudaEvent_t start, stop;\
	CHECK(cudaEventCreate(&start));\
	CHECK(cudaEventCreate(&stop))

#define rec_start CHECK(cudaEventRecord(start,0))

#define rec_stop CHECK(cudaEventRecord(stop,0));\
	CHECK(cudaEventSynchronize(stop));\
	cudaEventElapsedTime(&elapsedTime,start,stop)

#define rec_pr(s) printf(s" %3.6f ms\n",elapsedTime)

__device__ void sumReduce(float *dev_a,float *dev_sum);
__device__ void sumReduceLWD(float *dev_a,float *dev_sum);

void initialize(float *a);

#endif