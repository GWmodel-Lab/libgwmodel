#include "CudaUtils.h"
#include <iostream>
#include <armadillo>
using namespace std;
using namespace arma;

void pdm(const double* dptr, size_t rows, size_t cols, const char* header)
{
    double* mptr = new double[rows * cols];
    cudaMemcpy(mptr, dptr, sizeof(double) * rows * cols, cudaMemcpyDeviceToHost);
    mat tmp(mptr, rows, cols, false, true);
    tmp.brief_print(cout, header);
    delete[] mptr;
}