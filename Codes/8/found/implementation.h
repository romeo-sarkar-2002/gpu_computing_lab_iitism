#ifndef SUM_REDUCTION_H
#define SUM_REDUCTION_H

#include "stdlib.h"

#define BD 256
#define CHECK(call)                                                            \
  {                                                                            \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                   \
      fprintf(stderr, "code: %d, reason: %s\n", error,                         \
              cudaGetErrorString(error));                                      \
      exit(1);                                                                 \
    }                                                                          \
  }

struct Result {
  float ans;
  float time;
};

void atomic_tests(int *, int *);

struct Result kahan_sum(float *arr, size_t n);

void gpu_sort(float *arr, int n);

float khn_sum_host(float *arr, int n);

#endif
