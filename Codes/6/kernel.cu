#include <stdio.h>
#define N 8
#define TILE_WIDTH 2

__global__ void MatrixMulKernel (float *MatA, float *MatB, float *MatC, int Width)
{
	//Shared memory allocation
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;
	
	float Pvalue = 0;
	for(int ph = 0; ph < Width/TILE_WIDTH; ++ph)
    {
		//Collaborative loading of A and B tiles into shared memory
        Mds[ty][tx] = MatA[Row * Width + ph * TILE_WIDTH + tx];
        Nds[ty][tx] = MatB[(ph * TILE_WIDTH + ty) * Width + Col];
        __syncthreads();
        //dot product using shared memory
        for(int k = 0; k < TILE_WIDTH; ++k)
        {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    MatC[Row *Width+Col] = Pvalue;
}