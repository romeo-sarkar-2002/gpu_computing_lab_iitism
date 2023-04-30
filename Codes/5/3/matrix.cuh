#ifndef MATRIX_CUH
#define MATRIX_CUH
__global__ void initialize_GPU (double *p, int rows, int cols);
__global__ void addKernel (double *m1, double *m2, double *a, int rows, int cols);
__global__ void subKernel (double *m1, double *m2, double *a, int rows, int cols);
__global__ void mulKernel (double *m1, double *m2, double *p, int rows, int x, int cols);
__global__ void transposeKernel (double *m1, double *m2, int rows, int cols);

struct Matrix
{
    int rows, cols;
    double *device_pointer, *host_pointer;
    int flag = 0;
    
    Matrix ();
    Matrix (int r, int c);
    Matrix (const Matrix &M);
    Matrix (Matrix &&M);
    Matrix operator = (Matrix &M);
    Matrix operator = (Matrix &&M);
    ~Matrix ();
    void alloc ();
    void clear ();
    void display ();
    void init ();
    void H2D ();
    void D2H ();
    Matrix operator + (const Matrix &M);
    Matrix operator - (const Matrix &M);
    Matrix operator * (const Matrix &M);
    Matrix operator ~ ();
};

#endif