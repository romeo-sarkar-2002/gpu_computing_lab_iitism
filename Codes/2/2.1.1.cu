#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
__global__ void checkIndex ()
{
    printf ("device: \nthreadIdx: (%d, %d, %d)\nblockIdx: (%d, %d, %d)\nblockDim: (%d, %d, %d)\ngridDim: (%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
    return;
}
int main (int argc, char **argv)
{
    dim3 grid (256, 128, 1);
    dim3 block (1, 1, 1);
    printf ("host: \n");
    printf ("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
    printf ("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);
    checkIndex <<<grid, block>>> ();
    return 0;
}