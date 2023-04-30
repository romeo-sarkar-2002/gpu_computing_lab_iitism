#include <stdio.h>
void displayMatrix(float *A, int nx, int ny){
	int idx;
	for(int i = 0; i < nx; i++){
		for(int j = 0; j < ny; j++){
			idx = i*ny + j;
			printf(" %f ",A[idx]);
		}
	printf("\n");
	}
	return;
}
