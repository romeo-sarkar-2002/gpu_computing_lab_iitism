#include "implementation.h"
#include "stdio.h"
#include "stdlib.h"

void initialData(float *ip, const size_t size) {
  for (size_t i = 0; i < size; i++) {
    ip[i] = ((float)rand() / (float)(RAND_MAX));
  }
}

void print_arr(float *arr, const size_t size) {
  printf("[ ");
  for (size_t i = 0; i < size; i++) {
    printf("%f, ", arr[i]);
  }
  printf("]\n");
}

void compare_sum(float *arr, const size_t size) {
  struct Result gpu_ans = kahan_sum(arr, size);
  float host_ans = khn_sum_host(arr, size);
  printf("Device Sum = %f\n", gpu_ans.ans);
  printf("Host Sum = %f\n", host_ans);
  printf("Device Execution Time: %f\n", gpu_ans.time);
  if (host_ans == gpu_ans.ans) {
    printf("Host and GPU give have same accuracy\n");
  }
}

int main() {
  const int n = 1024;
  float *arr = (float *)malloc(n * sizeof(float));
  initialData(arr, n);

  printf("Unsorted Array Sum Comparison: \n");
  // print_arr(arr, n);
  compare_sum(arr, n);
  printf("\n");

  gpu_sort(arr, n);
  printf("Sorted Array Sum Comparison: \n");
  // print_arr(arr, n);
  compare_sum(arr, n);
  printf("\n");

  printf("Atomic Sum and Atomic Subtraction Test: \n");
  int var1 = 5;
  int var2 = 5;
  printf("Initial Var1 = %d\n", var1);
  printf("Initial Var2 = %d\n", var2);
  atomic_tests(&var1, &var2);
  printf("Final Var1 = %d\n", var1);
  printf("Final Var2 = %d\n", var2);

  free(arr);
}
