#ifndef BANDWIDTHWEIGHT_H
#define BANDWIDTHWEIGHT_H

#include <unordered_map>
#include <string>
#include "Weight.h"


namespace gwm
{

/**
 * @brief \~english Class for calculating weight with a bandwidth. 
 * Users can specific bandwidth size, bandwidth type and kernel function type.
 * 
 * There are two types of bandwidth: adaptive and fixed. 
 * If use an adaptive bandwidth, the value of bandwidth size \f$b\f$ must be integer, representing the distance to \f$b\f$-th nearest point. 
 * If use a fixed bandwidth, the value of bandwidth size is a distance.
 * 
 * There are five types of kernels: Gaussian, Exponential, Bisquare, Tricube and Boxcar.
 * Each type of kernel representing a kernel function. 
 * Users need only set the kernel type to let the instance call for the kernel function.
 * 
 * \~chinese 基于带宽计算权重的类。
 * 用户可以指定带宽的大小、类型和核函数类型。
 * 
 * 带宽的类型有两种：可变和固定。
 * 如果使用可变带宽，带宽大小 \f$b\f$ 必须是整数，代表到最近的第 \f$b\f$ 个点的距离。
 * 如果使用固定带宽，带宽大小是距离值。
 * 
 * 目前有五种核函数类型：Gaussian, Exponential, Bisquare, Tricube 和 Boxcar。
 * 每种类型的对应一种函数。
 * 用户只需要指定核函数类型以使实例调用对应的核函数。
 */
class BandwidthWeight : public Weight
{
public:

    /**
     * @brief \~english Type of kernel function. \~chinese 核函数类型。
     */
    enum KernelFunctionType
    {
        Gaussian,       //!< \~english Gaussian kernel BandwidthWeight::GaussianKernelFunction() \~chinese Gaussian 核函数 BandwidthWeight::GaussianKernelFunction()
        Exponential,    //!< \~english Exponential kernel BandwidthWeight::ExponentialKernelFunction() \~chinese Exponential 核函数 BandwidthWeight::ExponentialKernelFunction()
        Bisquare,       //!< \~english Bisquare kernel BandwidthWeight::BisquareKernelFunction() \~chinese Bisquare 核函数 BandwidthWeight::BisquareKernelFunction()
        Tricube,        //!< \~english Tricube kernel BandwidthWeight::TricubeKernelFunction() \~chinese Tricube 核函数 BandwidthWeight::TricubeKernelFunction()
        Boxcar          //!< \~english Boxcar kernel BandwidthWeight::BoxcarKernelFunction() \~chinese Boxcar 核函数 BandwidthWeight::BoxcarKernelFunction()
    };
    static std::unordered_map<KernelFunctionType, std::string> KernelFunctionTypeNameMapper;
    static std::unordered_map<bool, std::string> BandwidthTypeNameMapper;

    typedef arma::vec (*KernelFunction)(arma::vec, double); //!< \~english Kernel functions \~chinese 核函数

    static KernelFunction Kernel[];

    /**
     * @brief \~english Gaussian kernel function. \~chinese Gaussian 核函数。
     * 
     * @param dist \~english Distance vector \~chinese 距离向量
     * @param bw \~english Bandwidth size (its unit is equal to that of distance vector) \~chinese 带宽大小（和距离向量的单位相同）
     * @return \~english Weight value \~chinese 权重值
     */
    static arma::vec GaussianKernelFunction(arma::vec dist, double bw)
    {
        return exp((dist % dist) / ((-2.0) * (bw * bw)));
    }
    
    /**
     * @brief \~english Exponential kernel function. \~chinese Exponential 核函数。
     * 
     * @param dist \~english Distance vector \~chinese 距离向量
     * @param bw \~english Bandwidth size (its unit is equal to that of distance vector) \~chinese 带宽大小（和距离向量的单位相同）
     * @return \~english Weight value \~chinese 权重值
     */
    static arma::vec ExponentialKernelFunction(arma::vec dist, double bw)
    {
        return exp(-dist / bw);
    }
    
    /**
     * @brief \~english Bisquare kernel function. \~chinese Bisquare 核函数。
     * 
     * @param dist \~english Distance vector \~chinese 距离向量
     * @param bw \~english Bandwidth size (its unit is equal to that of distance vector) \~chinese 带宽大小（和距离向量的单位相同）
     * @return \~english Weight value \~chinese 权重值
     */
    static arma::vec BisquareKernelFunction(arma::vec dist, double bw)
    {
        arma::vec d2_d_b2 = 1.0 - (dist % dist) / (bw * bw);
        return (dist < bw) % (d2_d_b2 % d2_d_b2);
    }
    
    /**
     * @brief \~english Tricube kernel function. \~chinese Tricube 核函数。
     * 
     * @param dist \~english Distance vector \~chinese 距离向量
     * @param bw \~english Bandwidth size (its unit is equal to that of distance vector) \~chinese 带宽大小（和距离向量的单位相同）
     * @return \~english Weight value \~chinese 权重值
     */
    static arma::vec TricubeKernelFunction(arma::vec dist, double bw)
    {
        arma::vec d3_d_b3 = 1.0 - (dist % dist % dist) / (bw * bw * bw);
        return (dist < bw) % (d3_d_b3 % d3_d_b3 % d3_d_b3);
    }
    
    /**
     * @brief \~english Boxcar kernel function. \~chinese Boxcar 核函数。
     * 
     * @param dist \~english Distance vector \~chinese 距离向量
     * @param bw \~english Bandwidth size (its unit is equal to that of distance vector) \~chinese 带宽大小（和距离向量的单位相同）
     * @return \~english Weight value \~chinese 权重值
     */
    static arma::vec BoxcarKernelFunction(arma::vec dist, double bw)
    {
        return (dist < bw) % arma::vec(arma::size(dist), arma::fill::ones);
    }

public:

    /**
     * @brief \~english Construct a new BandwidthWeight object. \~chinese 构造一个新的 BandwidthWeight 对象。
     */
    BandwidthWeight() {}

    /**
     * @brief \~english Construct a new BandwidthWeight object. \~chinese 构造一个新的 BandwidthWeight 对象。
     * 
     * @param size \~english Bandwidth size \~chinese 带宽大小
     * @param adaptive \~english Whether use an adaptive bandwidth \~chinese 是否是可变带宽
     * @param kernel \~english Type of kernel function \~chinese 核函数类型
     */
    BandwidthWeight(double size, bool adaptive, KernelFunctionType kernel)
    {
        mBandwidth = size;
        mAdaptive = adaptive;
        mKernel = kernel;
    }

    /**
     * @brief \~english Copy construct a new BandwidthWeight object. \~chinese 复制构造一个 BandwidthWeight 对象。
     * 
     * @param bandwidthWeight \~english Reference to the object for copying \~chinese 要复制的对象引用
     */
    BandwidthWeight(const BandwidthWeight& bandwidthWeight)
    {
        mBandwidth = bandwidthWeight.mBandwidth;
        mAdaptive = bandwidthWeight.mAdaptive;
        mKernel = bandwidthWeight.mKernel;
    }

    /**
     * @brief \~english Copy construct a new BandwidthWeight object from a pointer. \~chinese 从指针复制构造一个 BandwidthWeight 对象。
     * 
     * @param bandwidthWeight \~english Pointer to the object for copying \~chinese 要复制的对象指针
     */
    BandwidthWeight(const BandwidthWeight* bandwidthWeight)
    {
        mBandwidth = bandwidthWeight->bandwidth();
        mAdaptive = bandwidthWeight->adaptive();
        mKernel = bandwidthWeight->kernel();
    }

    virtual Weight * clone() override
    {
        return new BandwidthWeight(*this);
    }

public:
    virtual arma::vec weight(arma::vec dist) override;

    /**
     * @brief \~english Get the bandwidth size. \~chinese 获取带宽大小。
     * 
     * @return \~english Bandwidth size \~chinese 带宽大小
     */
    double bandwidth() const
    {
        return mBandwidth;
    }

    /**
     * @brief \~english Set the bandwidth size. \~chinese 设置带宽大小。
     * 
     * @param bandwidth \~english Bandwidth size \~chinese 带宽大小
     */
    void setBandwidth(double bandwidth)
    {
        mBandwidth = bandwidth;
    }

    /**
     * @brief \~english Get whether it is adaptive bandwidth. \~chinese 获取是否使可变带宽。
     * 
     * @return true \~english Yes  \~chinese 是
     * @return false \~english No \~chinese 否
     */
    bool adaptive() const
    {
        return mAdaptive;
    }

    /**
     * @brief \~english Set whether it is adaptive bandwidth. \~chinese 设置是否使可变带宽。
     * 
     * @param bandwidth \~english Whether it is adaptive bandwidth \~chinese 是否使可变带宽
     */
    void setAdaptive(bool adaptive)
    {
        mAdaptive = adaptive;
    }

    /**
     * @brief \~english Get the type of kernel function. \~chinese 获取核函数类型。
     * 
     * @return KernelFunctionType \~english Type of kernel function \~chinese 核函数类型
     */
    KernelFunctionType kernel() const
    {
        return mKernel;
    }

    /**
     * @brief \~english Set the type of kernel function. \~chinese 设置核函数类型。
     * 
     * @param bandwidth \~english Type of kernel function \~chinese 核函数类型
     */
    void setKernel(const KernelFunctionType &kernel)
    {
        mKernel = kernel;
    }

private:
    double mBandwidth;          //!< \~english Bandwidth size \~chinese 带宽大小
    bool mAdaptive;             //!< \~english Whether it is adaptive bandwidth \~chinese 是否使可变带宽
    KernelFunctionType mKernel; //!< \~english Type of kernel function \~chinese 核函数类型
};

}

#endif // BANDWIDTHWEIGHT_H
