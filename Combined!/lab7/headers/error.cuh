#ifndef ERROR_CUH
#define ERROR_CUH

#define _ERROR_CHECK(foo) \
{ \
    cudaError_t err = (foo); \
    if (err != cudaSuccess) \
    { \
        printf ("%s(\033[1;32m%d\033[m): \033[1;4;31merror\033[m: \033[1;33m%s\033[m i.e. %s\n", __FILE__, __LINE__, cudaGetErrorName (err), cudaGetErrorString (err)); \
        exit (err); \
    } \
}


#endif