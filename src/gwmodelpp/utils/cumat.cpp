#include "cumat.hpp"

cublasHandle_t cumat::handle = nullptr;

cumat cumat::operator*(const cumat& right) const
{
    if (mCols != right.nrows()) throw std::logic_error("Dimension mismatched.");
    size_t m = mRows, k = mCols, n = right.ncols();
    cumat res {m, n};
    cublasStatus_t error = cublasDgemm(cumat::handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &cumat::alpha1, dMem, mRows, right.dmem(), right.nrows(), &cumat::beta0, res.dmem(), m);
    if (error != CUBLAS_STATUS_SUCCESS) throw error;
    return res;
}

cumat cumat::operator*(const cumat_trans& right) const
{
    if (mCols != right.nrows()) throw std::logic_error("Dimension mismatched.");
    size_t m = mRows, k = mCols, n = right.ncols();
    cumat res {m, n};
    cublasStatus_t error = cublasDgemm(cumat::handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &cumat::alpha1, dMem, mRows, right.mat.dmem(), right.mat.nrows(), &cumat::beta0, res.dmem(), m);
    if (error != CUBLAS_STATUS_SUCCESS) throw error;
    return res;
}

const cumat_trans cumat::t() const
{
    return cumat_trans(*this);
}

cumat cumat_trans::operator*(const cumat& right) const
{
    if (ncols() != right.nrows()) throw std::logic_error("Dimension mismatched.");
    size_t m = nrows(), k = ncols(), n = right.ncols();
    cumat res {m, n};
    cublasStatus_t error = cublasDgemm(cumat::handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &cumat::alpha1, mat.dmem(), mat.nrows(), right.dmem(), right.nrows(), &cumat::beta0, res.dmem(), m);
    if (error != CUBLAS_STATUS_SUCCESS) throw error;
    return res;
}

cumat cumat_trans::operator*(const cumat_trans& right) const
{
    if (ncols() != right.nrows()) throw std::logic_error("Dimension mismatched.");
    size_t m = nrows(), k = ncols(), n = right.ncols();
    cumat res {m, n};
    cublasStatus_t error = cublasDgemm(cumat::handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &cumat::alpha1, mat.dmem(), mat.nrows(), right.mat.dmem(), right.mat.nrows(), &cumat::beta0, res.dmem(), m);
    if (error != CUBLAS_STATUS_SUCCESS) throw error;
    return res;
}