#ifndef CUMAT_HPP
#define CUMAT_HPP

#include <exception>
#include <armadillo>
#include <cuda_runtime.h>
#include <cublas.h>

class cumat_trans;
class cumat;

class cumat
{
public:
    static cublasHandle_t handle = nullptr;
    static double alpha1 = 1.0;
    static double beta0 = 0.0;
    static double beta1 = 1.0;

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

    cumat(const cumat&& mat)
    {
        mRows = mat.mRows;
        mCols = mat.mCols;
        mBytes = mat.mBtypes;
        dMem = mat.dMem;
        mMoved = true;
    }

    ~cumat()
    {
        if (mMoved) cudaFree(dMem);
        dMem = nullptr;
        mRows = 0;
        mCols = 0;
        mBytes = 0;
    }

    const cumat_trans t() const
    {
        return cumat_trans(*this);
    }

    cumat operator*(const cumat& right) const
    {
        if (mCols != right.nrows()) throw std::logic_error("Dimension mismatched.");
        size_t m = mRows, k = mCols, n = right.ncols();
        cumat res {m, n};
        cudaError_t error = cublasDgemm(cumat::handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &cumat::alpha, dMem, mRows, right.dmem(), right.nrows(), &cumat::beta, res.dmem(), m);
        if (error != cudaSuccess) throw error;
        return std::move(res);
    }

    cumat operator*(const cumat_trans& right) const
    {
        if (mCols != right.nrows()) throw std::logic_error("Dimension mismatched.");
        size_t m = mRows, k = mCols, n = right.ncols();
        cumat res {m, n};
        cudaError_t error = cublasDgemm(cumat::handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &cumat::alpha, dMem, mRows, right.mat.dmem(), right.nrows(), &cumat::beta, res.dmem(), m);
        if (error != cudaSuccess) throw error;
        return std::move(res);
    }

public:
    size_t nrows() { return mRows; }
    size_t ncols() { return mCols; }
    size_t nbytes() { return mBytes; }
    double* dmem() { return dMem; }

private:
    bool mMoved = false;
    size_t mRows = 0;
    size_t mCols = 0;
    double* dMem = nullptr;
    size_t mBytes = 0;
}

class cumat_trans
{
public:
    cumat_trans(const cumat& src): mat(src) {}

    cumat_trans(const cumat_trans& src) : mat(src.mat) {}

    size_t nrows() const { return mat.ncols() }

    size_t ncols() const { return mat.nrows() }

    operator cumat()
    {
        size_t m = mat.ncols(), n = mat.nrows();
        cumat res { m, n };
        auto error = cublasDgeam(cublas::handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, &cumat::alpha1, mat.mem(), mat.nrows(), &cumat::beta0, mat.mem(), mat.nrows(), res.mem(), m);
        if (error != cudaSuccess) throw error;
        return std::move(res);
    }

    const cumat& t() const
    {
        return mat;
    }

    cumat operator*(const cumat& right) const
    {
        if (ncols() != right.nrows()) throw std::logic_error("Dimension mismatched.");
        size_t m = nrows(), k = ncols(), n = right.ncols();
        cumat res {m, n};
        cudaError_t error = cublasDgemm(cumat::handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &cumat::alpha, mat.dmem(), mat.nrows(), right.dmem(), right.nrows(), &cumat::beta, res.dmem(), m);
        if (error != cudaSuccess) throw error;
        return std::move(res);
    }

    cumat operator*(const cumat_trans& right) const
    {
        if (ncols() != right.nrows()) throw std::logic_error("Dimension mismatched.");
        size_t m = nrows(), k = ncols(), n = right.ncols();
        cumat res {m, n};
        cudaError_t error = cublasDgemm(cumat::handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &cumat::alpha, mat.dmem(), mat.nrows(), right.mat.dmem(), right.nrows(), &cumat::beta, res.dmem(), m);
        if (error != cudaSuccess) throw error;
        return std::move(res);
    }

    const cumat& mat;
}

#endif  // CUMAT_HPP