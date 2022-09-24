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

vec CGwmBandwidthWeight::weight(vec dist)
{
    const KernelFunction *kerf = Kernel + mKernel;
    uword nr = dist.n_elem;
    vec w(nr, fill::zeros);
    if (mAdaptive)
    {
        double dn = mBandwidth / nr, fixbw = 0;
        if (dn < 1)
        {
            vec vdist = sort(dist);
            double b0 = floor(mBandwidth), bx = mBandwidth - b0;
            double d0 = vdist(int(b0) - 1), d1 = vdist(int(b0));
            fixbw = d0 + (d1 - d0) * bx;
        }
        else
        {
            fixbw = dn * max(dist);
        }
        w = (*kerf)(dist, fixbw);
    }
    else
    {
        w = (*kerf)(dist, mBandwidth);
    }
    return w;
}
