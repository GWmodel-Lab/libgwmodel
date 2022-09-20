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
        if (dn <= 1)
        {
            vec vdist = sort(dist);
            fixbw = vdist(mBandwidth > 0 ? int(mBandwidth) - 1 : 0);
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
