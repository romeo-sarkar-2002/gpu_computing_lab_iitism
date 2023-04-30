#include<cuda_runtime.h>
#include<stdio.h>
#include "display.cuh"
#include "kernel.cuh"
#include "init.cuh"
#define N 8
#define TILE_WIDTH 2

int main(){
	//set up data size of matrix
	int Width = N;
	int nx = Width;
	int ny = Width;
	int nxy = nx * ny;
	int nBytes = nxy * sizeof(float);
	printf("Matrix size: %d by %d\n", nx, ny);
	printf("Tile size: %d by %d\n", TILE_WIDTH, TILE_WIDTH);
	
	//Malloc host memory
	float *h_A,*h_B,*h_C;
	h_A = (float *)malloc(nBytes);
	h_B = (float *)malloc(nBytes);
	h_C = (float *)malloc(nBytes);

	//initialize data at host side
	initialData(h_A,nxy);
	initialData(h_B,nxy);
	
	//Malloc device global memory
	float *d_MatA, *d_MatB, *d_MatC;
	cudaMalloc((void **)&d_MatA, nBytes);
	cudaMalloc((void **)&d_MatB, nBytes);
	cudaMalloc((void **)&d_MatC, nBytes);

	//transfer data from host to device
	cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);

	//invoke kernel at host side
	int bdimx = TILE_WIDTH;
	int bdimy = TILE_WIDTH;
	dim3 block(bdimx, bdimy);
	dim3 grid((nx + block.x - 1)/block.x, (ny + block.y - 1)/block.y);
	
	MatrixMulKernel<<<grid, block>>>(d_MatA, d_MatB, d_MatC, Width);
	cudaDeviceSynchronize();
	
	// copy kernel result back to host side
	cudaMemcpy(h_C, d_MatC, nBytes, cudaMemcpyDeviceToHost);
	printf("Matrix A is=\n");
	displayMatrix(h_A, nx, ny);
	printf("Matrix B is=\n");
	displayMatrix(h_B, nx, ny);
	printf("The product of Matrix A and Matrix B is=\n");
	displayMatrix(h_C, nx, ny);
	
	//free device global memory
	cudaFree(d_MatA);
	cudaFree(d_MatB);
	cudaFree(d_MatC);
	
	//free host memory
	free(h_A);
	free(h_B);
	free(h_C);
	
	//reset device
	cudaDeviceReset();
	
	return 0;
}