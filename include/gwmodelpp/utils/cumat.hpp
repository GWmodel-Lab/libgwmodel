#ifndef CUMAT_HPP
#define CUMAT_HPP

#include <exception>
#include <armadillo>
#include <cuda_runtime.h>
#include <cublas_v2.h>

class cumat_trans;
class cumat;

class cumat
{
public:
    static cublasHandle_t handle;
    constexpr static const double alpha1 = 1.0;
    constexpr static const double beta0 = 0.0;
    constexpr static const double beta1 = 1.0;

public:
    cumat(size_t rows, size_t cols) : mRows(rows), mCols(cols)
    {
        mBytes = sizeof(double) * rows * cols;
        cudaMalloc(&dMem, sizeof(double) * rows * cols);
        cudaMemset(dMem, 0, sizeof(double) * rows * cols);
    }

    cumat(arma::mat src) : cumat(src.n_rows, src.n_cols)
    {
        cudaMemcpy(dMem, src.mem, mBytes, cudaMemcpyHostToDevice);
    }

    cumat(const cumat& mat) : cumat(mat.mRows, mat.mCols)
    {
        cudaMemcpy(dMem, mat.dMem, mBytes, cudaMemcpyDeviceToDevice);
    }

    ~cumat()
    {
        cudaFree(dMem);
        dMem = nullptr;
        mRows = 0;
        mCols = 0;
        mBytes = 0;
    }

    const cumat_trans t() const;

    void get(double* dst)
    {
        cudaMemcpy(dst, dMem, mBytes, cudaMemcpyDeviceToHost);
    }

    cumat operator*(const cumat& right) const;

    cumat operator*(const cumat_trans& right) const;

public:
    size_t nrows() const { return mRows; }
    size_t ncols() const { return mCols; }
    size_t nbytes() const { return mBytes; }
    double* dmem() const { return dMem; }

private:
    size_t mRows = 0;
    size_t mCols = 0;
    double* dMem = nullptr;
    size_t mBytes = 0;
};

class cumat_trans
{
public:
    cumat_trans(const cumat& src): mat(src) {}

    cumat_trans(const cumat_trans& src) : mat(src.mat) {}

    size_t nrows() const { return mat.ncols(); }

    size_t ncols() const { return mat.nrows(); }

    operator cumat()
    {
        size_t m = mat.ncols(), n = mat.nrows();
        cumat res { m, n };
        auto error = cublasDgeam(cumat::handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, &cumat::alpha1, mat.dmem(), mat.nrows(), &cumat::beta0, mat.dmem(), mat.nrows(), res.dmem(), m);
        if (error != CUBLAS_STATUS_SUCCESS) throw error;
        return res;
    }

    const cumat& t() const
    {
        return mat;
    }

    cumat operator*(const cumat& right) const;

    cumat operator*(const cumat_trans& right) const;

    const cumat& mat;
};

#endif  // CUMAT_HPP