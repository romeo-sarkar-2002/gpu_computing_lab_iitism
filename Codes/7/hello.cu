#include <stdio.h>
__global__ void GPU ()
{
    printf ("hello world\n");
    return;
}
int main ()
{
    GPU <<< 2, 2 >>> ();
    return 0;
}