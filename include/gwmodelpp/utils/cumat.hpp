#ifndef CUMAT_HPP
#define CUMAT_HPP

#include <exception>
#include <armadillo>
#include <cuda_runtime.h>
#include <cublas_v2.h>

class cubase;
class cumat;
class custride;

template<class C, class A, class B>
C cumatmul(const A&& a, const B&& b);

template<class T>
struct cutraits
{
    cubase::Type type = T::type;
    cuop::Op op = T::op;
};

class cuop
{
public:
    enum class Op
    {
        Origin = CUBLAS_OP_N,
        Trans = CUBLAS_OP_T
    };
};

class cubase
{
public:
    static cublasHandle_t handle;
    constexpr static const double alpha1 = 1.0;
    constexpr static const double beta0 = 0.0;
    constexpr static const double beta1 = 1.0;

    enum class Type
    {
        Base,
        Op,
        Mat,
        Stride,
    };

    cuop::Op op = cuop::Op::Origin;
    cubase::Type type = cubase::Type::Base;

public:
    cubase() {}

    cubase(size_t bytes)
    {
        cudaMalloc(&dMem, bytes);
        cudaMemset(dMem, 0, bytes);
    }

    virtual ~cubase()
    {
        if (dMem) cudaFree(dMem);
        mBytes = 0;
    }

    size_t nbytes() const { return mBytes; }
    double* dmem() const { return dMem; }

protected:
    double* dMem = nullptr;
    size_t mBytes = 0;
};

template<typename T>
class cuop_trans : public cuop
{
public:
    cuop::Op op = cuop::Op::Trans;
    cubase::Type type = T::type;

public:
    cuop_trans(const T& src): ori(src) {}

    cuop_trans(const cuop_trans& src) : ori(src.mat) {}

    size_t nrows() const { return ori.ncols(); }
    size_t ncols() const { return ori.nrows(); }
    double* dmem() const { return ori.dmem(); }

    T operator*(const T& right) const;
    T operator*(const cuop_trans<T>& right) const;

    template<typename R, typename V>
    V operator*(const R& right) const
    {
        throw std::logic_error("Not implemented");
    }

    template<typename R, typename V>
    V operator*(const cuop_trans<R>& right) const
    {
        throw std::logic_error("Not implemented");
    }

    const T& ori;
};

class cumat : public cubase
{
public:
    cubase::Type type = cubase::Type::Mat;

public:
    cumat() {}

    cumat(size_t rows, size_t cols) : 
        cubase(rows * cols),
        mRows(rows),
        mCols(cols)
    {
    }

    cumat(arma::mat src) : cumat(src.n_rows, src.n_cols)
    {
        cudaMemcpy(dMem, src.mem, mBytes, cudaMemcpyHostToDevice);
    }

    cumat(const cumat& mat) : cumat(mat.mRows, mat.mCols)
    {
        cudaMemcpy(dMem, mat.dMem, mBytes, cudaMemcpyDeviceToDevice);
    }

    virtual ~cumat()
    {
        mRows = 0;
        mCols = 0;
        mBytes = 0;
    }

    const cuop_trans<cumat> t() const
    {
        return cuop_trans<cumat>(*this);
    }

    void get(double* dst)
    {
        cudaMemcpy(dst, dMem, mBytes, cudaMemcpyDeviceToHost);
    }

    cumat& operator=(const cuop_trans<cumat>& right)
    {
        size_t m = right.ori.ncols(), n = right.ori.nrows();
        cumat res { m, n };
        auto error = cublasDgeam(
            cubase::handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, 
            &cubase::alpha1, right.ori.dmem(), right.ori.nrows(), 
            &cubase::beta0, right.ori.dmem(), right.ori.nrows(),
            res.dmem(), m
        );
        if (error != CUBLAS_STATUS_SUCCESS) throw error;
        return res;
    }

    cumat operator*(const cumat& right) const;
    cumat operator*(const cuop_trans<cumat>& right) const;

    custride operator*(const custride& right) const;
    custride operator*(const cuop_trans<custride>& right) const;


public:
    size_t nrows() const { return mRows; }
    size_t ncols() const { return mCols; }

protected:
    size_t mRows = 0;
    size_t mCols = 0;
};

class custride: public cubase
{
public:
    cubase::Type type = cubase::Type::Stride;

public:
    custride(size_t rows, size_t cols, size_t strides) : 
        cubase(sizeof(double) * rows * cols * strides),
        mRows(rows),
        mCols(cols),
        mStrides(strides),
        mStrideBytes(sizeof(double) * rows * cols)
    {
        cudaMalloc(&dMem, mBytes);
        cudaMemset(dMem, 0, mBytes);
    }

    virtual ~custride()
    {
        mRows = 0;
        mCols = 0;
        mBytes = 0;
        mStrides = 0;
        mStrideBytes = 0;
        cudaFree(dMem);
    }

    size_t nrows() const { return mRows; }
    size_t ncols() const { return mCols; }
    size_t nstrides() const { return mStrides; }
    size_t nstrideBytes() const { return mStrideBytes; }

    custride operator*(const cumat& right) const;
    custride operator*(const custride& right) const;
    custride operator*(const cuop_trans<cumat>& right) const;
    custride operator*(const cuop_trans<custride>& right) const;

protected:
    size_t mRows = 0;
    size_t mCols = 0;
    size_t mStrides = 0;
    size_t mStrideBytes = 0;
};

template<class C, class A, class B>
C cumatmul(const A&& a, const B&& b)
{
    size_t m = a.nrows(), k = a.ncols(), n = b.ncols();
    int lda = cutraits<A>::Op == cuop::Op::Origin ? a.nrows() : a.ncols();
    int ldb = cutraits<B>::Op == cuop::Op::Origin ? b.nrows() : b.ncols();
    // if either l or r is stride matrix
    if (cutraits<A>::type == cubase::Type::Stride || cutraits<B>::type == cubase::Type::Stride)
    {
        int strideA = cutraits<A>::Type == cubase::Type::Stride ? a.nstrides() : 0;
        int strideB = cutraits<B>::Type == cubase::Type::Stride ? b.nstrides() : 0;
        int strideC = cutraits<A>::Type == cubase::Type::Stride ? strideA : (cutraits<B>::Type == cubase::Type::Stride ? strideB : 0);
        custride c { m, n, strideC };
        cublasStatus_t error = cublasDgemmStridedBatched(
            cubase::handle, cutraits<A>::op, cutraits<B>::op,
            m, n, k, &cubase::alpha1,
            a.dmem(), lda, strideA,
            b.dmem(), ldb, strideB,
            &cubase::beta0, c.dmem(), m, strideC, strideC
        );
        if (error != CUBLAS_STATUS_SUCCESS) throw error;
        return c;
    }
    else
    {
        cumat c { m, n };
        cublasStatus_t error = cublasDgemm(
            cubase::handle, cutraits<A>::op, cutraits<B>::op,
            m, n, k, &cubase::alpha1,
            a.dmem(), lda,
            b.dmem(), ldb,
            &cubase::beta0, c.dmem(), m
        );
        if (error != CUBLAS_STATUS_SUCCESS) throw error;
        return c;
    }
}

#endif  // CUMAT_HPP