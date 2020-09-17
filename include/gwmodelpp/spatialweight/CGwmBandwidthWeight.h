#ifndef CGWMBANDWIDTHWEIGHT_H
#define CGWMBANDWIDTHWEIGHT_H

#include <unordered_map>
#include <string>
#include "gwmodelpp.h"
#include "spatialweight/CGwmWeight.h"

using namespace std;

class GWMODELPP_API CGwmBandwidthWeight : public CGwmWeight
{
public:
    enum KernelFunctionType
    {
        Gaussian,
        Exponential,
        Bisquare,
        Tricube,
        Boxcar
    };
    static unordered_map<KernelFunctionType, string> KernelFunctionTypeNameMapper;
    static unordered_map<bool, string> BandwidthTypeNameMapper;

    typedef double (*KernelFunction)(double, double);

    static KernelFunction Kernel[];

    static double GaussianKernelFunction(double dist, double bw);
    static double ExponentialKernelFunction(double dist, double bw);
    static double BisquareKernelFunction(double dist, double bw);
    static double TricubeKernelFunction(double dist, double bw);
    static double BoxcarKernelFunction(double dist, double bw);

public:
    CGwmBandwidthWeight();
    CGwmBandwidthWeight(double size, bool adaptive, KernelFunctionType kernel);
    CGwmBandwidthWeight(const CGwmBandwidthWeight& bandwidthWeight);
    CGwmBandwidthWeight(const CGwmBandwidthWeight* bandwidthWeight);

    virtual CGwmWeight * clone() override
    {
        return new CGwmBandwidthWeight(*this);
    }

public:
    virtual vec weight(vec dist) override;

    double bandwidth() const;
    void setBandwidth(double bandwidth);

    bool adaptive() const;
    void setAdaptive(bool adaptive);

    KernelFunctionType kernel() const;
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
