#ifndef CUMAT_HPP
#define CUMAT_HPP

#include <exception>
#include <armadillo>
#include <cuda_runtime.h>
#include <cublas_v2.h>

class cubase;
class cumat;
class custride;
template <class T>
class cuop_trans;

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

    constexpr static cuop::Op op = cuop::Op::Origin;
    constexpr static cubase::Type type = cubase::Type::Base;

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
    }

    virtual size_t nbytes() const = 0;

    virtual void get(double* dst)
    {
        cudaMemcpy(dst, dMem, nbytes(), cudaMemcpyDeviceToHost);
    }

    double* dmem() const { return dMem; }

protected:
    double* dMem = nullptr;
};

template<class T>
struct cutraits
{
    constexpr static cubase::Type type = T::type;
    constexpr static cuop::Op op = T::op;
};

template<class A, class B, cubase::Type TA, cubase::Type TB>
class cuop_matmul;

class cumat : public cubase
{
public:
    constexpr static cuop::Op op = cuop::Op::Origin;
    constexpr static cubase::Type type = cubase::Type::Mat;

public:
    cumat() {}

    cumat(size_t rows, size_t cols) : 
        cubase(rows * cols * sizeof(double)),
        mRows(rows),
        mCols(cols)
    {
    }

    cumat(arma::mat src) : cumat(src.n_rows, src.n_cols)
    {
        cudaMemcpy(dMem, src.mem, nbytes(), cudaMemcpyHostToDevice);
    }

    cumat(const cumat& mat) : cumat(mat.mRows, mat.mCols)
    {
        cudaMemcpy(dMem, mat.dMem, nbytes(), cudaMemcpyDeviceToDevice);
    }

    virtual ~cumat()
    {
        mRows = 0;
        mCols = 0;
    }

    size_t nbytes() const override { return sizeof(double) * mRows * mCols; }

    const cuop_trans<cumat> t() const;

    cumat& operator=(const cuop_trans<cumat>& right);

    template<class R>
    auto operator*(const R& right) const
    {
        return cuop_matmul<cumat, R, cumat::type, cutraits<R>::type>(*this, right).eval();
    }


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
    constexpr static cuop::Op op = cuop::Op::Origin;
    constexpr static cubase::Type type = cubase::Type::Stride;

public:
    custride() {}

    custride(size_t rows, size_t cols, size_t strides) : 
        cubase(sizeof(double) * rows * cols * strides),
        mRows(rows),
        mCols(cols),
        mStrides(strides)
    {
        cudaMalloc(&dMem, nbytes());
        cudaMemset(dMem, 0, nbytes());
    }

    custride(const arma::cube& src):
        cubase(sizeof(double) * src.n_elem),
        mRows(src.n_rows),
        mCols(src.n_cols),
        mStrides(src.n_slices)
    {
        cudaMemcpy(dMem, src.mem, nbytes(), cudaMemcpyHostToDevice);
    }

    custride(const custride& mat) : custride(mat.mRows, mat.mCols, mat.mStrides)
    {
        cudaMemcpy(dMem, mat.dMem, nbytes(), cudaMemcpyDeviceToDevice);
    }

    size_t nbytes() const override { return sizeof(double) * mRows * mCols * mStrides; }

    virtual ~custride()
    {
        mRows = 0;
        mCols = 0;
        mStrides = 0;
        cudaFree(dMem);
    }

    size_t nrows() const { return mRows; }
    size_t ncols() const { return mCols; }
    size_t nstrides() const { return mStrides; }
    size_t nstrideBytes() const { return mRows * mCols * sizeof(double); }

    const cuop_trans<custride> t() const;

    template<class R>
    auto operator*(const R& right) const
    {
        return cuop_matmul<custride, R, custride::type, cutraits<R>::type>(*this, right).eval();
    }

protected:
    size_t mRows = 0;
    size_t mCols = 0;
    size_t mStrides = 0;
};

template<typename T>
class cuop_trans : public cuop
{
public:
    constexpr static cuop::Op op = cuop::Op::Trans;
    constexpr static cubase::Type type = cutraits<T>::type;

public:
    cuop_trans(const T& src): ori(src) {}

    cuop_trans(const cuop_trans& src) : ori(src.mat) {}

    size_t nrows() const { return ori.ncols(); }
    size_t ncols() const { return ori.nrows(); }
    double* dmem() const { return ori.dmem(); }

    template<class R>
    auto operator*(const R& right) const
    {
        return cuop_matmul<cuop_trans<T>, R, cutraits<T>::type, cutraits<R>::type>(*this, right).eval();
    }

    const T& ori;
};


template<class A, class B, cubase::Type TA, cubase::Type TB>
class cuop_matmul
{
public:
    cuop_matmul(const A& left, const B& right): a(left), b(right) {}

    template<class T>
    int getStrides(const T& m)
    {
        return 1;
    }

    template<class T>
    int getStrideSize(const T& m)
    {
        return cutraits<T>::type == cubase::Type::Stride ? m.nrows() * m.ncols() : 0;
    }

    int getStrides(const custride& m)
    {
        return m.nstrides();
    }

    int getStrides(const cuop_trans<custride>& m)
    {
        return m.ori.nstrides();
    }

    auto eval()
    {
        size_t m = a.nrows(), k = a.ncols(), n = b.ncols();
        int lda = cutraits<A>::op == cuop::Op::Origin ? a.nrows() : a.ncols();
        int ldb = cutraits<B>::op == cuop::Op::Origin ? b.nrows() : b.ncols();
        auto opa = cutraits<A>::op == cuop::Op::Origin ? CUBLAS_OP_N : CUBLAS_OP_T;
        auto opb = cutraits<B>::op == cuop::Op::Origin ? CUBLAS_OP_N : CUBLAS_OP_T;
        size_t strideSizeA = getStrideSize(a);
        size_t strideSizeB = getStrideSize(b);
        size_t strideC = cutraits<A>::type == cubase::Type::Stride ? getStrides(a) : (cutraits<B>::type == cubase::Type::Stride ? getStrides(b) : 1);
        custride c { m, n, strideC };
        int strideSizeC = getStrideSize(c);
        cublasStatus_t error = cublasDgemmStridedBatched(
            cubase::handle, opa, opb,
            m, n, k, &cubase::alpha1,
            a.dmem(), lda, strideSizeA,
            b.dmem(), ldb, strideSizeB,
            &cubase::beta0, c.dmem(), m, strideSizeC, strideC
        );
        if (error != CUBLAS_STATUS_SUCCESS) throw cublasGetStatusString(error);
        return c;
    }

private:
    const A& a;
    const B& b;
};

template<class A, class B>
class cuop_matmul<A, B, cubase::Type::Mat, cubase::Type::Mat>
{
public:

    cuop_matmul(const A& left, const B& right): a(left), b(right) {}

    auto eval()
    {
        size_t m = a.nrows(), k = a.ncols(), n = b.ncols();
        int lda = cutraits<A>::op == cuop::Op::Origin ? a.nrows() : a.ncols();
        int ldb = cutraits<B>::op == cuop::Op::Origin ? b.nrows() : b.ncols();
        auto opa = cutraits<A>::op == cuop::Op::Origin ? CUBLAS_OP_N : CUBLAS_OP_T;
        auto opb = cutraits<B>::op == cuop::Op::Origin ? CUBLAS_OP_N : CUBLAS_OP_T;
        cumat c { m, n };
        cublasStatus_t error = cublasDgemm(
            cubase::handle, opa, opb,
            m, n, k, &cubase::alpha1,
            a.dmem(), lda,
            b.dmem(), ldb,
            &cubase::beta0, c.dmem(), m
        );
        if (error != CUBLAS_STATUS_SUCCESS) throw cublasGetStatusString(error);
        return c;
    }

private:
    const A& a;
    const B& b;
};

#endif  // CUMAT_HPP