#include "cumat.hpp"

cublasHandle_t cubase::handle = nullptr;

cumat cumat::operator*(const cumat& right) const
{
    size_t m = mRows, k = mCols, n = right.ncols();
    cumat res {m, n};
    cublasStatus_t error = cublasDgemm(cubase::handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &cubase::alpha1, dMem, mRows, right.dmem(), right.nrows(), &cubase::beta0, res.dmem(), m);
    if (error != CUBLAS_STATUS_SUCCESS) throw error;
    return res;
}

cumat cumat::operator*(const cuop_trans<cumat>& right) const
{
    size_t m = mRows, k = mCols, n = right.ncols();
    cumat res {m, n};
    cublasStatus_t error = cublasDgemm(cubase::handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &cubase::alpha1, dMem, mRows, right.mat.dmem(), right.mat.nrows(), &cubase::beta0, res.dmem(), m);
    if (error != CUBLAS_STATUS_SUCCESS) throw error;
    return res;
}

custride cumat::operator*(const custride& right) const
{
    size_t m = nrows(), k = ncols(), n = right.ncols();
    custride res {m, n, right.nstrides()};
    cublasStatus_t error = cublasDgemmStridedBatched(
        cubase::handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &cubase::alpha1,
        dMem, nrows(), 0,
        right.dmem(), right.nrows(), right.nstrideBytes(),
        &cubase::beta0, res.dmem(), res.nrows(), res.nstrideBytes(),
        right.nstrides()
    );
    if (error != CUBLAS_STATUS_SUCCESS) throw error;
    return res;
}

custride cumat::operator*(const cuop_trans<custride>& right) const
{
    size_t m = nrows(), k = ncols(), n = right.ncols();
    custride res {m, n, right.ori.nstrides()};
    cublasStatus_t error = cublasDgemmStridedBatched(
        cubase::handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k,
        &cubase::alpha1, dMem, nrows(), 0,
        right.ori.dmem(), right.nrows(), right.ori.nstrideBytes(),
        &cubase::beta0, res.dmem(), res.nrows(), res.nstrideBytes(),
        right.ori.nstrides()
    );
    if (error != CUBLAS_STATUS_SUCCESS) throw error;
    return res;
}

cumat cuop_trans<cumat>::operator*(const cumat& right) const
{
    size_t m = nrows(), k = ncols(), n = right.ncols();
    cumat res {m, n};
    cublasStatus_t error = cublasDgemm(cubase::handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &cubase::alpha1, mat.dmem(), mat.nrows(), right.dmem(), right.nrows(), &cubase::beta0, res.dmem(), m);
    if (error != CUBLAS_STATUS_SUCCESS) throw error;
    return res;
}

cumat cuop_trans<cumat>::operator*(const cuop_trans<cumat>& right) const
{
    size_t m = nrows(), k = ncols(), n = right.ncols();
    cumat res {m, n};
    cublasStatus_t error = cublasDgemm(cubase::handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &cubase::alpha1, mat.dmem(), mat.nrows(), right.mat.dmem(), right.mat.nrows(), &cubase::beta0, res.dmem(), m);
    if (error != CUBLAS_STATUS_SUCCESS) throw error;
    return res;
}

custride custride::operator*(const cumat& right) const
{
    size_t m = nrows(), k = ncols(), n = right.ncols();
    custride res {m, n, nstrides()};
    cublasStatus_t error = cublasDgemmStridedBatched(
        cubase::handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &cubase::alpha1,
        dMem, nrows(), nstrideBytes(),
        right.dmem(), right.nrows(), 0,
        &cubase::beta0, res.dmem(), res.nrows(), res.nstrideBytes(),
        nstrides()
    );
    if (error != CUBLAS_STATUS_SUCCESS) throw error;
    return res;
}

custride custride::operator*(const cuop_trans<cumat>& right) const
{
    size_t m = nrows(), k = ncols(), n = right.ncols();
    custride res {m, n, nstrides()};
    cublasStatus_t error = cublasDgemmStridedBatched(
        cubase::handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &cubase::alpha1,
        dMem, nrows(), nstrideBytes(),
        right.ori.dmem(), right.ori.nrows(), 0,
        &cubase::beta0, res.dmem(), res.nrows(), res.nstrideBytes(),
        nstrides()
    );
    if (error != CUBLAS_STATUS_SUCCESS) throw error;
    return res;
}

custride custride::operator*(const custride& right) const
{
    size_t m = nrows(), k = ncols(), n = right.ncols();
    custride res {m, n, nstrides()};
    cublasStatus_t error = cublasDgemmStridedBatched(
        cubase::handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &cubase::alpha1,
        dMem, nrows(), nstrideBytes(),
        right.dmem(), right.nrows(), right.nstrideBytes(),
        &cubase::beta0, res.dmem(), res.nrows(), res.nstrideBytes(),
        nstrides()
    );
    if (error != CUBLAS_STATUS_SUCCESS) throw error;
    return res;
}

custride custride::operator*(const cuop_trans<custride>& right) const
{
    size_t m = nrows(), k = ncols(), n = right.ncols();
    custride res {m, n, nstrides()};
    cublasStatus_t error = cublasDgemmStridedBatched(
        cubase::handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &cubase::alpha1,
        dMem, nrows(), nstrideBytes(),
        right.ori.dmem(), right.nrows(), right.ori.nstrideBytes(),
        &cubase::beta0, res.dmem(), res.nrows(), res.nstrideBytes(),
        nstrides()
    );
    if (error != CUBLAS_STATUS_SUCCESS) throw error;
    return res;
}
