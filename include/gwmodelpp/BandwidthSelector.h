#ifndef BANDWIDTHSELECTOR_H
#define BANDWIDTHSELECTOR_H

#include <map>
#include <vector>
#include <utility>
#include "IBandwidthSelectable.h"
#include "spatialweight/BandwidthWeight.h"

namespace gwm
{

class BandwidthSelector
{
public:
    BandwidthSelector() {}
    BandwidthSelector(BandwidthWeight* bandwidth, double lower, double upper) : mBandwidth(bandwidth) , mLower(lower) , mUpper(upper) {}
    ~BandwidthSelector() {}

public:
    BandwidthWeight *bandwidth() const;
    void setBandwidth(BandwidthWeight *bandwidth);

    double lower() const;
    void setLower(double lower);

    double upper() const;
    void setUpper(double upper);

    BandwidthCriterionList bandwidthCriterion() const;

public:
    BandwidthWeight* optimize(IBandwidthSelectable* instance);

private:
    BandwidthWeight* mBandwidth;
    double mLower;
    double mUpper;
    std::unordered_map<double, double> mBandwidthCriterion;
};

inline BandwidthWeight *BandwidthSelector::bandwidth() const
{
    return mBandwidth;
}

inline void BandwidthSelector::setBandwidth(BandwidthWeight *bandwidth)
{
    mBandwidth = bandwidth;
}

inline double BandwidthSelector::lower() const
{
    return mLower;
}

inline void BandwidthSelector::setLower(double lower)
{
    mLower = lower;
}

inline double BandwidthSelector::upper() const
{
    return mUpper;
}

inline void BandwidthSelector::setUpper(double upper)
{
    mUpper = upper;
}

}

#endif  // BANDWIDTHSELECTOR_H