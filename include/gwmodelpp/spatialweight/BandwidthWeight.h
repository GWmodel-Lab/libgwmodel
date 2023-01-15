#ifndef BANDWIDTHWEIGHT_H
#define BANDWIDTHWEIGHT_H

#include <unordered_map>
#include <string>
#include "Weight.h"


namespace gwm
{

/**
 * @brief Class for calculating weight with a bandwidth.
 * Users can specific bandwidth size, bandwidth type and kernel function type.
 * 
 * There are two types of bandwidth: adaptive and fixed. 
 * If use an adaptive bandwidth, the value of bandwidth size \f$b\f$ must be integer, representing the distance to \f$b\f$-th nearest point. 
 * If use a fixed bandwidth, the value of bandwidth size is a distance.
 * 
 * There are five types of kernels: Gaussian, Exponential, Bisquare, Tricube and Boxcar.
 * Each type of kernel representing a kernel function. 
 * Users need only set the kernel type to let the instance call for the kernel function.
 */
class BandwidthWeight : public Weight
{
public:

    /**
     * @brief Type of kernel function.
     */
    enum KernelFunctionType
    {
        Gaussian,       // Call for gaussian kernel BandwidthWeight::GaussianKernelFunction().
        Exponential,    // Call for exponential kernel BandwidthWeight::ExponentialKernelFunction().
        Bisquare,       // Call for bisquare kernel BandwidthWeight::BisquareKernelFunction().
        Tricube,        // Call for tricube kernel BandwidthWeight::TricubeKernelFunction().
        Boxcar          // Call for boxcar kernel BandwidthWeight::BoxcarKernelFunction().
    };
    static std::unordered_map<KernelFunctionType, std::string> KernelFunctionTypeNameMapper;
    static std::unordered_map<bool, std::string> BandwidthTypeNameMapper;

    typedef arma::vec (*KernelFunction)(arma::vec, double);

    static KernelFunction Kernel[];

    /**
     * @brief Gaussian kernel function.
     * 
     * @param dist Distance vector. 
     * @param bw Bandwidth size. The unit is equal to that of distance vector.
     * @return Weight value.
     */
    static arma::vec GaussianKernelFunction(arma::vec dist, double bw)
    {
        return exp((dist % dist) / ((-2.0) * (bw * bw)));
    }
    
    /**
     * @brief Exponential kernel function.
     * 
     * @param dist Distance vector. 
     * @param bw Bandwidth size. The unit is equal to that of distance vector.
     * @return Weight value.
     */
    static arma::vec ExponentialKernelFunction(arma::vec dist, double bw)
    {
        return exp(-dist / bw);
    }
    
    /**
     * @brief Bisquare kernel function.
     * 
     * @param dist Distance vector. 
     * @param bw Bandwidth size. The unit is equal to that of distance vector.
     * @return Weight value.
     */
    static arma::vec BisquareKernelFunction(arma::vec dist, double bw)
    {
        arma::vec d2_d_b2 = 1.0 - (dist % dist) / (bw * bw);
        return (dist < bw) % (d2_d_b2 % d2_d_b2);
    }
    
    /**
     * @brief Tricube kernel function.
     * 
     * @param dist Distance vector. 
     * @param bw Bandwidth size. The unit is equal to that of distance vector.
     * @return Weight value.
     */
    static arma::vec TricubeKernelFunction(arma::vec dist, double bw)
    {
        arma::vec d3_d_b3 = 1.0 - (dist % dist % dist) / (bw * bw * bw);
        return (dist < bw) % (d3_d_b3 % d3_d_b3 % d3_d_b3);
    }
    
    /**
     * @brief Boxcar kernel function.
     * 
     * @param dist Distance vector. 
     * @param bw Bandwidth size. The unit is equal to that of distance vector.
     * @return Weight value.
     */
    static arma::vec BoxcarKernelFunction(arma::vec dist, double bw)
    {
        return (dist < bw) % arma::vec(arma::size(dist), arma::fill::ones);
    }

public:

    /**
     * @brief Construct a new BandwidthWeight object.
     */
    BandwidthWeight() {}

    /**
     * @brief Construct a new BandwidthWeight object.
     * 
     * @param size Bandwidth size. 
     * @param adaptive Whether use an adaptive bandwidth. 
     * @param kernel Type of kernel function.
     */
    BandwidthWeight(double size, bool adaptive, KernelFunctionType kernel)
    {
        mBandwidth = size;
        mAdaptive = adaptive;
        mKernel = kernel;
    }

    /**
     * @brief Construct a new BandwidthWeight object.
     * 
     * @param bandwidthWeight Reference to the object for copying.
     */
    BandwidthWeight(const BandwidthWeight& bandwidthWeight)
    {
        mBandwidth = bandwidthWeight.mBandwidth;
        mAdaptive = bandwidthWeight.mAdaptive;
        mKernel = bandwidthWeight.mKernel;
    }

    /**
     * @brief Construct a new BandwidthWeight object.
     * 
     * @param bandwidthWeight Pointer to the object for copying.
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
     * @brief Get the BandwidthWeight::mBandwidth object.
     * 
     * @return Bandwidth size. 
     */
    double bandwidth() const
    {
        return mBandwidth;
    }

    /**
     * @brief Set the BandwidthWeight::mBandwidth object.
     * 
     * @param bandwidth Bandwidth size. 
     */
    void setBandwidth(double bandwidth)
    {
        mBandwidth = bandwidth;
    }

    /**
     * @brief Get the BandwidthWeight::mAdaptive object.
     * 
     * @return true if use an adaptive bandwidth. 
     * @return false if use an fixed bandwidth.
     */
    bool adaptive() const
    {
        return mAdaptive;
    }

    /**
     * @brief Set the BandwidthWeight::mAdaptive object.
     * 
     * @param bandwidth Whether use an adaptive bandwidth. 
     */
    void setAdaptive(bool adaptive)
    {
        mAdaptive = adaptive;
    }

    /**
     * @brief Get the BandwidthWeight::mKernel object.
     * 
     * @return Type of kernel function. 
     */
    KernelFunctionType kernel() const
    {
        return mKernel;
    }

    /**
     * @brief Set the BandwidthWeight::mBandwidth object.
     * 
     * @param bandwidth Type of kernel function. 
     */
    void setKernel(const KernelFunctionType &kernel)
    {
        mKernel = kernel;
    }

private:
    double mBandwidth;
    bool mAdaptive;
    KernelFunctionType mKernel;
};

}

#endif // BANDWIDTHWEIGHT_H
