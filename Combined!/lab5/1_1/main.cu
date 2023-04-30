#include <stdio.h>
#include "Matrix.cuh"
int main ()
{
    srand (time (NULL));

    Matrix M1 (5, 5), M2 (5, 5);
    M1.initialize (), M2.initialize ();
    
    Matrix Sum = M1 + M2;
    Matrix Sum2 = Sum * Sum;

    DISPLAY (M1);
    DISPLAY (M2);
    DISPLAY (Sum);
    DISPLAY (Sum2);

    cudaDeviceReset ();
    return 0;
}