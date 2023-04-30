#include <stdio.h>
#include <time.h>
void initialize_array (long long *arr, size_t size)
{
    struct timespec start, stop;
    timespec_get (&start, TIME_UTC);
    for (size_t i = 0; i < size; i++)
    {
        // arr[i] = (long long) rand ();
        arr[i] = 1LL;
    }
    timespec_get (&stop, TIME_UTC);
    printf ("\033[90mtime taken to initialize the array: %.9lf secs.\033[m\n", ((double) (stop.tv_nsec - start.tv_nsec) * 1e-9 + ((double) (stop.tv_sec - start.tv_sec))));
    return;
}
