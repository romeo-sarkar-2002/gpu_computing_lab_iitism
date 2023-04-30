__global__ void init_GPU (double *p, int rows, int cols);
__global__ void mul_GPU (double *m1, double *m2, double *p, int rows, int x, int cols);
__global__ void trp_GPU (double *m1, double *m2, int rows, int cols);
__global__ void add_GPU (double *m1, double *m2, double *a, int rows, int cols);
__global__ void sub_GPU (double *m1, double *m2, double *a, int rows, int cols);
void init (double *p, int rows, int cols);
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