#ifndef WHATEVER_H_INCLUDED
#define WHATEVER_H_INCLUDED
class matrix{
    public:
    int rows, cols;
    double *host_pointer, *device_pointer;
    matrix();
	matrix(int r, int c);
    ~matrix();
    void memAllocInBoth();
    void display();
    void init_rand();
    void H2D();
	void D2H();
};
// extern "cuda"
// {
__device__ void add_matrix_GPU(double *d_MatA ,double *d_MatB ,double *d_MatC, int rows, int cols);
__global__ void mul_matrix_GPU(double *d_MatA ,double *d_MatB ,double *d_MatC, int rows, int width, int cols);
// }
#endif