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
        Batched
    };

    constexpr static cuop::Op op = cuop::Op::Origin;
    constexpr static cubase::Type type = cubase::Type::Base;

    enum class Init
    {
        None,
        Zero
    };

public:
    cubase() {}

    cubase(size_t bytes, Init init = Init::Zero)
    {
        switch (init)
        {
        case Init::Zero:
            mIsRelease = true;
            cudaMalloc(&dMem, bytes);
            cudaMemset(dMem, 0, bytes);
            break;
        default:
            break;
        }
    }

    virtual ~cubase()
    {
        if (mIsRelease && dMem)
        {
            cudaFree(dMem);
        }
        dMem = nullptr;
    }

    virtual size_t nbytes() const = 0;

    virtual void get(double* dst)
    {
        cudaMemcpy(dst, dMem, nbytes(), cudaMemcpyDeviceToHost);
    }

    double* dmem() const { return dMem; }

protected:
    bool mIsRelease = false;
    double* dMem = nullptr;
};

template<class T>
struct cutraits
{
    constexpr static cubase::Type type = T::type;
    constexpr static cuop::Op op = T::op;
};

template<class A, class B, cubase::Type TA = cutraits<A>::type, cubase::Type TB = cutraits<B>::type>
class cuop_matmul;

class cumat : public cubase
{
public:
    constexpr static cuop::Op op = cuop::Op::Origin;
    constexpr static cubase::Type type = cubase::Type::Mat;

public:
    cumat() {}

    cumat(size_t rows, size_t cols, cubase::Init init = cubase::Init::Zero) : 
        cubase(rows * cols * sizeof(double), init),
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

    cumat(cumat&& mat) : mRows(mat.mRows), mCols(mat.mCols)
    {
        mIsRelease = true;
        dMem = mat.dMem;
        mat.mIsRelease = false;
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
        return cuop_matmul<cumat, R>(*this, right).eval();
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

    custride(size_t rows, size_t cols, size_t strides, cubase::Init init = cubase::Init::Zero) : 
        cubase(sizeof(double) * rows * cols * strides, init),
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

    custride(custride&& mat) : mRows(mat.mRows), mCols(mat.mCols), mStrides(mat.mStrides)
    {
        mIsRelease = true;
        dMem = mat.dMem;
        mat.mIsRelease = false;
    }

    size_t nbytes() const override { return sizeof(double) * mRows * mCols * mStrides; }

    virtual ~custride()
    {
        mRows = 0;
        mCols = 0;
        mStrides = 0;
    }

    size_t nrows() const { return mRows; }
    size_t ncols() const { return mCols; }
    size_t nstrides() const { return mStrides; }
    size_t nstrideSize() const { return mRows * mCols; }
    size_t nstrideBytes() const { return mRows * mCols * sizeof(double); }

    const cuop_trans<custride> t() const;

    custride inv(int* d_info) const;

    template<class R>
    auto operator*(const R& right) const
    {
        return cuop_matmul<custride, R>(*this, right).eval();
    }

protected:
    size_t mRows = 0;
    size_t mCols = 0;
    size_t mStrides = 0;
};

class cubatched
{
public:
    constexpr static cuop::Op op = cuop::Op::Origin;
    constexpr static cubase::Type type = cubase::Type::Batched;

public:
    cubatched() {}

    cubatched(size_t batch) : mBatch(batch)
    {
        cudaMalloc(&dArray, sizeof(double*) * mBatch);
        cudaMemset(dArray, 0, sizeof(double*) * mBatch);
    }

    cubatched(size_t batch, size_t rows, size_t cols, bool initialize = false) : cubatched(batch)
    {
        mRows = rows;
        mCols = cols;
        if (initialize)
        {
            mIsRelease = true;
            for (size_t i = 0; i < batch; i++)
            {
                cudaMalloc(dArray + i, sizeof(double) * rows * cols);
                cudaMemset(dArray[i], 0, sizeof(double) * rows * cols);
            }
        }
    }

    cubatched(const cubatched& right) : cubatched(right.mBatch, right.mRows, right.mCols, true)
    {
        for (size_t i = 0; i < mBatch; i++)
        {
            cudaMemcpy(dArray[i], right.dArray[i], sizeof(double) * mRows * mCols, cudaMemcpyDeviceToDevice);
        }
    }

    cubatched(cubatched&& right) : cubatched(right.mBatch, right.mRows, right.mCols, false)
    {
        dArray = right.dArray;
        right.mIsRelease = false;
    }

    cubatched(const double** p_mats, size_t batch, size_t rows, size_t cols) : cubatched(batch, rows, cols)
    {
        cudaMemcpy(dArray, p_mats, sizeof(double*) * batch, cudaMemcpyHostToDevice);        
    }

    cubatched(double* d_mat, size_t batch, size_t rows, size_t cols, size_t bias) : cubatched(batch, rows, cols)
    {
        double** p_mats = new double*[mBatch];
        for (size_t i = 0; i < mBatch; i++)
        {
            p_mats[i] = d_mat + i * bias;
        }
        cudaMemcpy(dArray, p_mats, sizeof(double*) * mBatch, cudaMemcpyHostToDevice);
        delete[] p_mats;
    }

    cubatched(const custride& stride) : cubatched(stride.dmem(), stride.nstrides(), stride.nrows(), stride.ncols(), stride.nstrideSize())
    {}

    cubatched(const cumat& mat) : cubatched(mat.dmem(), mat.ncols(), mat.nrows(), mat.nrows(), 1)
    {}

    cubatched(const cumat& mat, size_t batch) : cubatched(mat.dmem(), batch, 0, mat.nrows(), mat.ncols())
    {}

    ~cubatched()
    {
        if (mIsRelease && dArray)
        {
            for (size_t i = 0; i < mBatch; i++)
            {
                cudaFree(dArray[i]);
            }
        }
        cudaFree(dArray);
        dArray = nullptr;
    }

    double** darray() { return dArray; }
    size_t nrows() { return mRows; }
    size_t ncols() { return mCols; }
    size_t nbatch() { return mBatch; }

public:
    cubatched inv(int* dinfo)
    {
        cubatched res(mBatch, mRows, mCols, true);
        cublasDmatinvBatched(cubase::handle, mRows, dArray, mRows, res.darray(), mRows, dinfo, mBatch);
        return res;
    }

private:
    size_t mBatch = 0;
    double** dArray = nullptr;
    size_t mRows = 0;
    size_t mCols = 0;
    bool mIsRelease = false;
};

template<typename T>
class cuop_trans : public cuop
{
public:
    constexpr static cuop::Op op = cuop::Op::Trans;
    constexpr static cubase::Type type = cutraits<T>::type;

public:
    cuop_trans(const T& src): ori(src) {}

    cuop_trans(const cuop_trans& src) : ori(src.ori) {}

    double* dmem() const { return ori.dmem(); }
    size_t nrows() const { return ori.ncols(); }
    size_t ncols() const { return ori.nrows(); }

    template<class R>
    auto operator*(const R& right) const
    {
        return cuop_matmul<cuop_trans<T>, R>(*this, right).eval();
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
        return std::move(c);
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
        return std::move(c);
    }

private:
    const A& a;
    const B& b;
};

#endif  // CUMAT_HPP