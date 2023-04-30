#ifndef ERROR_H
#define ERROR_H
#define CHECK(foo)\
{\
    enum cudaError err = foo;\
    if (err != cudaSuccess)\
    {\
        printf (__FILE__ ":" "[\033[1;33m%d\033[m]: " #foo ";\n\033[1;4;31m%s\033[0;1;31m:\033[m %s", __LINE__, cudaGetErrorName (err), cudaGetErrorString (err));\
        exit (err);\
    }\
}

#endif
