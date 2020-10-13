#include "gwmodelpp/spatialweight/CGwmBandwidthWeight.h"
#include "gwmodelpp/spatialweight/CGwmWeight.h"

unordered_map<CGwmBandwidthWeight::KernelFunctionType, string> CGwmBandwidthWeight::KernelFunctionTypeNameMapper = {
    std::make_pair(CGwmBandwidthWeight::KernelFunctionType::Boxcar, "Boxcar"),
    std::make_pair(CGwmBandwidthWeight::KernelFunctionType::Tricube, "Tricube"),
    std::make_pair(CGwmBandwidthWeight::KernelFunctionType::Bisquare, "Bisquare"),
    std::make_pair(CGwmBandwidthWeight::KernelFunctionType::Gaussian, "Gaussian"),
    std::make_pair(CGwmBandwidthWeight::KernelFunctionType::Exponential, "Exponential")
};

unordered_map<bool, string> CGwmBandwidthWeight::BandwidthTypeNameMapper = {
    std::make_pair(true, "Adaptive"),
    std::make_pair(false, "Fixed")
};

CGwmBandwidthWeight::KernelFunction CGwmBandwidthWeight::Kernel[] =
{
    &CGwmBandwidthWeight::GaussianKernelFunction,
    &CGwmBandwidthWeight::ExponentialKernelFunction,
    &CGwmBandwidthWeight::BisquareKernelFunction,
    &CGwmBandwidthWeight::TricubeKernelFunction,
    &CGwmBandwidthWeight::BoxcarKernelFunction
};

CGwmBandwidthWeight::CGwmBandwidthWeight() : CGwmWeight()
{

}

CGwmBandwidthWeight::CGwmBandwidthWeight(double size, bool adaptive, CGwmBandwidthWeight::KernelFunctionType kernel)
{
    mBandwidth = size;
    mAdaptive = adaptive;
    mKernel = kernel;
}

CGwmBandwidthWeight::CGwmBandwidthWeight(const CGwmBandwidthWeight &bandwidthWeight)
{
    mBandwidth = bandwidthWeight.mBandwidth;
    mAdaptive = bandwidthWeight.mAdaptive;
    mKernel = bandwidthWeight.mKernel;
}

CGwmBandwidthWeight::CGwmBandwidthWeight(const CGwmBandwidthWeight *bandwidthWeight)
{
    mBandwidth = bandwidthWeight->bandwidth();
    mAdaptive = bandwidthWeight->adaptive();
    mKernel = bandwidthWeight->kernel();
}

vec CGwmBandwidthWeight::weight(vec dist)
{
    const KernelFunction *kerf = Kernel + mKernel;
    uword nr = dist.n_elem;
    vec w(nr, fill::zeros);
    if (mAdaptive)
    {
        double dn = mBandwidth / nr, fixbw = 0;
        if (dn <= 1)
        {
            vec vdist = sort(dist);
            fixbw = vdist(mBandwidth > 0 ? int(mBandwidth) - 1 : 0);
        }
        else
        {
            fixbw = dn * max(dist);
        }
        for (uword r = 0; r < nr; r++)
        {
            w(r) = (*kerf)(dist(r), fixbw);
        }
    }
    else
    {
        for (uword r = 0; r < nr; r++)
        {
            w(r) = (*kerf)(dist(r), mBandwidth);
        }
    }
    return w;
}
