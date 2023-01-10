#include "gwmodelpp/spatialweight/BandwidthWeight.h"
#include "gwmodelpp/spatialweight/Weight.h"

using namespace std;
using namespace arma;
using namespace gwm;

unordered_map<BandwidthWeight::KernelFunctionType, string> BandwidthWeight::KernelFunctionTypeNameMapper = {
    std::make_pair(BandwidthWeight::KernelFunctionType::Boxcar, "Boxcar"),
    std::make_pair(BandwidthWeight::KernelFunctionType::Tricube, "Tricube"),
    std::make_pair(BandwidthWeight::KernelFunctionType::Bisquare, "Bisquare"),
    std::make_pair(BandwidthWeight::KernelFunctionType::Gaussian, "Gaussian"),
    std::make_pair(BandwidthWeight::KernelFunctionType::Exponential, "Exponential")
};

unordered_map<bool, string> BandwidthWeight::BandwidthTypeNameMapper = {
    std::make_pair(true, "Adaptive"),
    std::make_pair(false, "Fixed")
};

BandwidthWeight::KernelFunction BandwidthWeight::Kernel[] =
{
    &BandwidthWeight::GaussianKernelFunction,
    &BandwidthWeight::ExponentialKernelFunction,
    &BandwidthWeight::BisquareKernelFunction,
    &BandwidthWeight::TricubeKernelFunction,
    &BandwidthWeight::BoxcarKernelFunction
};

vec BandwidthWeight::weight(vec dist)
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
