#include <stdio.h>
#include "Matrix.cuh"
int main ()
{
    srand (time (NULL));
    Matrix M1 (3, 5), M2 (3, 5);
    M1.init (), M2.init ();
    Matrix Sum = M1 + M2;
    printf ("Matrix M1:\n");
    M1.display ();
    printf ("Matrix M2:\n");
    M2.display ();
    printf ("Matrix Sum:\n");
    Sum.display ();
    cudaDeviceSynchronize ();
    return 0;
}