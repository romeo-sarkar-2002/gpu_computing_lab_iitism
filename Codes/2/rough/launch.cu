#include <stdio.h>
__global__ void foo ()
{
    printf ("threadIdx: (%d, %d, %d)\nblockIdx: (%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.y, blockIdx.x, blockIdx.y, blockIdx.z);
    return;
}
int main ()
{
    dim3 grid (2, 2, 2), block (1, 1, 1);
    foo <<<grid, block>>> ();
    return 0;
}