#include <stdio.h>
void initialData (float *ip, const int size)
{
	for(int i = 0; i < size; i++) ip[i] = ((float) rand()/(float)(RAND_MAX));
	return;
}