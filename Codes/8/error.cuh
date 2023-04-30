#ifndef ERROR_CUH
#define ERROR_CUH

#define chkError(param) \
{ \
    cudaError_t err = (param); \
    if (err != cudaSuccess) \
    { \
        printf ("%s(\033[1;32m%d\033[m): \033[1;4;31merror\033[m: \033[1;33m%s\033[m i.e. %s\n", __FILE__, __LINE__, cudaGetErrorName (err), cudaGetErrorString (err)); \
        exit (err); \
    } \
}
#define getLastError() \
{ \
    cudaError_t err = cudaGetLastError (); \
    if (err != cudaSuccess) \
    { \
        printf ("%s(\033[1;32m%d\033[m): \033[1;4;31merror\033[m: \033[1;33m%s\033[m i.e. %s\n", __FILE__, __LINE__, cudaGetErrorName (err), cudaGetErrorString (err)); \
        exit (err); \
    } \
}

#endif