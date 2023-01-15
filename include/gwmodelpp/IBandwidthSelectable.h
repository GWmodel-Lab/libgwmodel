#ifndef IBANDWIDTHSELECTABLE_H
#define IBANDWIDTHSELECTABLE_H

#include "spatialweight/BandwidthWeight.h"

namespace gwm
{

struct IBandwidthSelectable
{
    virtual double getCriterion(BandwidthWeight* weight) = 0;
};

typedef std::vector<std::pair<double, double> >  BandwidthCriterionList;


}

#endif  // IBANDWIDTHSELECTABLE_H