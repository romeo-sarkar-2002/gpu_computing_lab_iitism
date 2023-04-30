#ifndef MATRIX_CUH
#define MATRIX_CUH

#include <stdio.h>

#define DISPLAY(x) \
printf ("\033[4;31mMatrix " #x ":\033[m\n");\
x.display ();

void initialize (double *p, int rows, int cols);

struct Matrix
{
    int rows, cols;
    double *device_pointer, *host_pointer;
    
    Matrix ();
    Matrix (int r, int c);
    Matrix (const Matrix &M);
    Matrix (Matrix &&M);

    Matrix operator = (Matrix &M);
    Matrix operator = (Matrix &&M);

    ~Matrix ();

    void initialize ();
    void display ();
    void alloc ();
    void clear ();
    void H2D ();
    void D2H ();

    Matrix operator * (const Matrix &M);
    Matrix operator + (const Matrix &M);
};

#endif