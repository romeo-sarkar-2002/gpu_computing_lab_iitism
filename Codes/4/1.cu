#include <stdio.h>
#include <cuda_runtime.h>
#define N 3
__global__ void MatrixMulKernel (float *MatA, float *MatB, float *MatC, int Width)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    if (Row < Width && Col < Width)
    {
        // printf ("{%d,%d}", Row, Col);
        float Pvalue = 0;
        for (int k = 0; k < Width; k++)
        {
            // printf ("(%.0f,%.0f)", MatA[Row * Width + k], MatB[k * Width + Col]);
            Pvalue += MatA[Row * Width + k] * MatB[k * Width + Col];
        }
        MatC[Row * Width + Col] = Pvalue;
        // printf ("=<%f>\n", Pvalue);
    }
}
void initialData (float *ip, const int size)
{
    // int i;
    for (int i = 0; i < size; i++)
    {
        ip[i] = i;
    }
}
void displayMatrix (float *A, int nx, int ny, int widthField)
{
    int idx;
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            idx = i * ny + j;
            printf (" %*.0f ", widthField, A[idx]);
        }
        printf ("\n");
    }
}
int main ()
{
    int Width = N;
    int nx = Width;
    int ny = Width;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof (float);
    printf ("Matrix size: nx %d ny %d\n", nx, ny);
    
    float *h_A, *h_B, *h_C;
    h_A = (float *) (malloc (nBytes));
    h_B = (float *) malloc (nBytes);
    h_C = (float *) malloc (nBytes);
    
    initialData (h_A, nxy);
    initialData (h_B, nxy);
    
    float *d_MatA, *d_MatB, *d_MatC;
    cudaMalloc ((void **) &d_MatA, nBytes);
    cudaMalloc ((void **) &d_MatB, nBytes);
    cudaMalloc ((void **) &d_MatC, nBytes);


    cudaMemcpy ((void *) d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy ((void *) d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);

    int bdimx = 16;
    int bdimy = 16;

    dim3 block (bdimx, bdimy, 1);
    dim3 grid ((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y, 1);

    MatrixMulKernel <<<grid, block>>> (d_MatA, d_MatB, d_MatC, Width);
    cudaDeviceSynchronize ();

    cudaMemcpy (h_C, d_MatC, nBytes, cudaMemcpyDeviceToHost);

    printf ("Matrix A is=\n");
    displayMatrix (h_A, nx, ny, 2);
    printf ("Matrix B is=\n");
    displayMatrix (h_B, nx, ny, 2);
    printf ("The Product of Matrix A and Matrix B is=\n");
    displayMatrix (h_C, nx, ny, 5);

    cudaFree (d_MatA);
    cudaFree (d_MatB);
    cudaFree (d_MatC);

    free (h_A);
    free (h_B);
    free (h_C);

    cudaDeviceReset ();
    return 0;

}