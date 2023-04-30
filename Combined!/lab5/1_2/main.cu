#include <stdio.h>
#include "matrix.cuh"
int main ()
{
    srand (time (NULL));

    Matrix A (4, 3);
    A.initialize ();

    Matrix AT = ~A;
    Matrix P = A * AT;

    DISPLAY (A);
    DISPLAY (AT);
    DISPLAY (P);

    cudaDeviceReset ();
    return 0;
}