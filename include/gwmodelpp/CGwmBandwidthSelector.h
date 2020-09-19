#ifndef CGWMBANDWIDTHSELECTOR_H
#define CGWMBANDWIDTHSELECTOR_H

#include <map>
#include <vector>
#include <utility>
#include "IGwmBandwidthSelectable.h"
#include "spatialweight/CGwmBandwidthWeight.h"

using namespace std;

class CGwmBandwidthSelector
{
public:
    CGwmBandwidthSelector();
    CGwmBandwidthSelector(CGwmBandwidthWeight* bandwidth, double lower, double upper);
    ~CGwmBandwidthSelector();

public:
    CGwmBandwidthWeight *bandwidth() const;
    void setBandwidth(CGwmBandwidthWeight *bandwidth);

    double lower() const;
    void setLower(double lower);

    double upper() const;
    void setUpper(double upper);

    BandwidthCriterionList bandwidthCriterion() const;

public:
    CGwmBandwidthWeight* optimize(IGwmBandwidthSelectable* instance);

private:
    CGwmBandwidthWeight* mBandwidth;
    double mLower;
    double mUpper;
    unordered_map<double, double> mBandwidthCriterion;
};

inline CGwmBandwidthWeight *CGwmBandwidthSelector::bandwidth() const
{
    return mBandwidth;
}

inline void CGwmBandwidthSelector::setBandwidth(CGwmBandwidthWeight *bandwidth)
{
    mBandwidth = bandwidth;
}

inline double CGwmBandwidthSelector::lower() const
{
    return mLower;
}

inline void CGwmBandwidthSelector::setLower(double lower)
{
    mLower = lower;
}

inline double CGwmBandwidthSelector::upper() const
{
    return mUpper;
}

inline void CGwmBandwidthSelector::setUpper(double upper)
{
    mUpper = upper;
}

#endif  // CGWMBANDWIDTHSELECTOR_H