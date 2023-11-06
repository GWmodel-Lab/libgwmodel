#include "cumat.hpp"

cublasHandle_t cubase::handle = nullptr;

const cuop_trans<cumat> cumat::t() const
{
    return cuop_trans<cumat>(*this);
}

custride cumat::as_stride() const
{
    return custride(*this);
}

cumat cumat::diagmul(const cumat& diag) const
{
    cumat res { mRows, mCols };
    cublasDdgmm(
        cubase::handle, CUBLAS_SIDE_RIGHT, mRows, mCols, 
        dMem, mRows, 
        diag.dmem(), 1, 
        res.dmem(), mRows
    );
    return res;
}

cuview<custride> custride::strides(size_t start) const
{
    curange range { start, start + 1 };
    cuview<custride> view(*this, range);
    return view;
}

cuview<custride> custride::strides(size_t start, size_t end) const
{
    curange range { start, end };
    cuview<custride> view(*this, range);
    return view;
}

const cuop_trans<custride> custride::t() const
{
    return cuop_trans<custride>(*this);
}

custride custride::inv(int* d_info) const
{
    custride d_inv(mRows, mCols, mStrides);
    cubatched b_array(*this), b_inv(d_inv);
    cublasDmatinvBatched(cubase::handle, mRows, b_array.darray(), b_array.nrows(), b_inv.darray(), b_inv.nrows(), d_info, b_array.nbatch());
    return d_inv;
}

cumat& cumat::operator=(const cuop_trans<cumat>& right)
{
    mRows = right.ori.ncols(), mCols = right.ori.nrows();
    if (dMem) cudaFree(dMem);
    cudaMalloc(&dMem, nbytes());
    auto error = cublasDgeam(
        cubase::handle, CUBLAS_OP_T, CUBLAS_OP_T, mRows, mCols, 
        &cubase::alpha1, right.ori.dmem(), right.ori.nrows(), 
        &cubase::beta0, right.ori.dmem(), right.ori.nrows(),
        dMem, mRows
    );
    if (error != CUBLAS_STATUS_SUCCESS) throw error;
    return *this;
}
