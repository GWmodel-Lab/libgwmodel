#include "CudaUtils.h"
#include <iostream>
#include <armadillo>
using namespace std;
using namespace arma;

void pdm(const double* dptr, size_t rows, size_t cols, const char* header)
{
    mat tmp(rows, cols);
    cudaMemcpy(tmp.memptr(), dptr, sizeof(double) * rows * cols, cudaMemcpyDeviceToHost);
    tmp.brief_print(cout, header);
}

void pdc(const double* dptr, size_t rows, size_t cols, size_t strides, const char* header)
{
    cube tmp(rows, cols, strides);
    cudaMemcpy(tmp.memptr(), dptr, sizeof(double) * rows * cols * strides, cudaMemcpyDeviceToHost);
    tmp.brief_print(cout, header);
}
