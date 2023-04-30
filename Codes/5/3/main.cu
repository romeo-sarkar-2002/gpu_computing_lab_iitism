#include <stdio.h>
#include "matrix.cuh"
int main ()
{
    srand (time (NULL));

    Matrix A (4, 3);
    A.initialize ();
    Matrix AT = ~A;
    printf ("Matrix A:\n");
    A.display ();
    printf ("Matrix AT:\n");
    AT.display ();
    Matrix P = A * AT;
    printf ("Matrix P:\n");
    P.display ();
    cudaDeviceReset ();
    return 0;
}