// #include <stdio.h>
#define last_error_check()\
    cudaError_t err = cudaGetLastError (); \
    if (err != cudaSuccess || 1) \
    { \
        printf ("%s(%d): \033[1;31merror\033[m: %s i.e. %s\n", __FILE__, __LINE__, cudaGetErrorString (err), cudaGetErrorName (err));\
        exit err; \
    }
last_error_check ()