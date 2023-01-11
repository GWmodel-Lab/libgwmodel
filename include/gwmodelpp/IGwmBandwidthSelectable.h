#ifndef IGWMBANDWIDTHSELECTABLE_H
#define IGWMBANDWIDTHSELECTABLE_H

#include "spatialweight/CGwmBandwidthWeight.h"

struct IGwmBandwidthSelectable
{
    virtual double getCriterion(CGwmBandwidthWeight* weight) = 0;
};

typedef std::vector<std::pair<double, double> >  BandwidthCriterionList;


#endif  // IGWMBANDWIDTHSELECTABLE_H