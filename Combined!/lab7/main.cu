#include <stdio.h>
// #include "headers/error.cuh"
// #include "headers/macro.cuh"
#include "headers/sumReduction.cuh"
#include "headers/sumReduction_lessWarpDivergence.cuh"
#include "headers/arrayManip.cuh"

int main ()
{
    srand (time (NULL));
    size_t size = 256578645; // array size;
    long long *arr = (long long *) (malloc (sizeof (long long) * size));
    initialize_array (arr, size);

    printf ("\033[4mreducedSum:\033[m %lld\n", sum (arr, size));
    printf ("\033[4mreducedSum_lessWarpDivergence:\033[m %lld\n", sum_lessWarpDivergence (arr, size));
    
    return 0;
}