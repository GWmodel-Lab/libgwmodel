#ifndef CGWMBANDWIDTHWEIGHT_H
#define CGWMBANDWIDTHWEIGHT_H

#include <unordered_map>
#include <string>
#include "gwmodelpp/spatialweight/CGwmWeight.h"

using namespace std;

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
class CGwmBandwidthWeight : public CGwmWeight
{
public:

    /**
     * @brief Type of kernel function.
     */
    enum KernelFunctionType
    {
        Gaussian,       // Call for gaussian kernel CGwmBandwidthWeight::GaussianKernelFunction().
        Exponential,    // Call for exponential kernel CGwmBandwidthWeight::ExponentialKernelFunction().
        Bisquare,       // Call for bisquare kernel CGwmBandwidthWeight::BisquareKernelFunction().
        Tricube,        // Call for tricube kernel CGwmBandwidthWeight::TricubeKernelFunction().
        Boxcar          // Call for boxcar kernel CGwmBandwidthWeight::BoxcarKernelFunction().
    };
    static unordered_map<KernelFunctionType, string> KernelFunctionTypeNameMapper;
    static unordered_map<bool, string> BandwidthTypeNameMapper;

    typedef double (*KernelFunction)(double, double);

    static KernelFunction Kernel[];

    /**
     * @brief Gaussian kernel function.
     * 
     * @param dist Distance vector. 
     * @param bw Bandwidth size. The unit is equal to that of distance vector.
     * @return Weight value.
     */
    static double GaussianKernelFunction(double dist, double bw);
    
    /**
     * @brief Exponential kernel function.
     * 
     * @param dist Distance vector. 
     * @param bw Bandwidth size. The unit is equal to that of distance vector.
     * @return Weight value.
     */
    static double ExponentialKernelFunction(double dist, double bw);
    
    /**
     * @brief Bisquare kernel function.
     * 
     * @param dist Distance vector. 
     * @param bw Bandwidth size. The unit is equal to that of distance vector.
     * @return Weight value.
     */
    static double BisquareKernelFunction(double dist, double bw);
    
    /**
     * @brief Tricube kernel function.
     * 
     * @param dist Distance vector. 
     * @param bw Bandwidth size. The unit is equal to that of distance vector.
     * @return Weight value.
     */
    static double TricubeKernelFunction(double dist, double bw);
    
    /**
     * @brief Boxcar kernel function.
     * 
     * @param dist Distance vector. 
     * @param bw Bandwidth size. The unit is equal to that of distance vector.
     * @return Weight value.
     */
    static double BoxcarKernelFunction(double dist, double bw);

public:

    /**
     * @brief Construct a new CGwmBandwidthWeight object.
     */
    CGwmBandwidthWeight();

    /**
     * @brief Construct a new CGwmBandwidthWeight object.
     * 
     * @param size Bandwidth size. 
     * @param adaptive Whether use an adaptive bandwidth. 
     * @param kernel Type of kernel function.
     */
    CGwmBandwidthWeight(double size, bool adaptive, KernelFunctionType kernel);

    /**
     * @brief Construct a new CGwmBandwidthWeight object.
     * 
     * @param bandwidthWeight Reference to the object for copying.
     */
    CGwmBandwidthWeight(const CGwmBandwidthWeight& bandwidthWeight);

    /**
     * @brief Construct a new CGwmBandwidthWeight object.
     * 
     * @param bandwidthWeight Pointer to the object for copying.
     */
    CGwmBandwidthWeight(const CGwmBandwidthWeight* bandwidthWeight);

    virtual CGwmWeight * clone() override
    {
        return new CGwmBandwidthWeight(*this);
    }

public:
    virtual vec weight(vec dist) override;

    /**
     * @brief Get the CGwmBandwidthWeight::mBandwidth object.
     * 
     * @return Bandwidth size. 
     */
    double bandwidth() const;

    /**
     * @brief Set the CGwmBandwidthWeight::mBandwidth object.
     * 
     * @param bandwidth Bandwidth size. 
     */
    void setBandwidth(double bandwidth);

    /**
     * @brief Get the CGwmBandwidthWeight::mAdaptive object.
     * 
     * @return true if use an adaptive bandwidth. 
     * @return false if use an fixed bandwidth.
     */
    bool adaptive() const;

    /**
     * @brief Set the CGwmBandwidthWeight::mAdaptive object.
     * 
     * @param bandwidth Whether use an adaptive bandwidth. 
     */
    void setAdaptive(bool adaptive);

    /**
     * @brief Get the CGwmBandwidthWeight::mKernel object.
     * 
     * @return Type of kernel function. 
     */
    KernelFunctionType kernel() const;

    /**
     * @brief Set the CGwmBandwidthWeight::mBandwidth object.
     * 
     * @param bandwidth Type of kernel function. 
     */
    void setKernel(const KernelFunctionType &kernel);

private:
    double mBandwidth;
    bool mAdaptive;
    KernelFunctionType mKernel;
};

inline double CGwmBandwidthWeight::GaussianKernelFunction(double dist, double bw) {
  return exp((dist * dist)/((-2)*(bw * bw)));
}

inline double CGwmBandwidthWeight::ExponentialKernelFunction(double dist, double bw) {
  return exp(-dist/bw);
}

inline double CGwmBandwidthWeight::BisquareKernelFunction(double dist, double bw) {
  return dist > bw ? 0 : (1 - (dist * dist)/(bw * bw)) * (1 - (dist * dist)/(bw * bw));
}

inline double CGwmBandwidthWeight::TricubeKernelFunction(double dist, double bw) {
  return dist > bw ?
              0 :
              (1 - (dist * dist * dist)/(bw * bw * bw)) *
              (1 - (dist * dist * dist)/(bw * bw * bw)) *
              (1 - (dist * dist * dist)/(bw * bw * bw));
}

inline double CGwmBandwidthWeight::BoxcarKernelFunction(double dist, double bw) {
  return dist > bw ? 0 : 1;
}

inline double CGwmBandwidthWeight::bandwidth() const
{
    return mBandwidth;
}

inline void CGwmBandwidthWeight::setBandwidth(double bandwidth)
{
    mBandwidth = bandwidth;
}

inline bool CGwmBandwidthWeight::adaptive() const
{
    return mAdaptive;
}

inline void CGwmBandwidthWeight::setAdaptive(bool adaptive)
{
    mAdaptive = adaptive;
}

inline CGwmBandwidthWeight::KernelFunctionType CGwmBandwidthWeight::kernel() const
{
    return mKernel;
}

inline void CGwmBandwidthWeight::setKernel(const KernelFunctionType &kernel)
{
    mKernel = kernel;
}

#endif // CGWMBANDWIDTHWEIGHT_H
