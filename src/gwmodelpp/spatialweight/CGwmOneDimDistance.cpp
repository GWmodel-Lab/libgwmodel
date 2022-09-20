#include "gwmodelpp/spatialweight/CGwmOneDimDistance.h"
#include <assert.h>

#include <exception>


CGwmOneDimDistance::CGwmOneDimDistance() : CGwmDistance()
{

}

CGwmOneDimDistance::CGwmOneDimDistance(const CGwmOneDimDistance &distance) : CGwmDistance(distance)
{
    
}

DistanceParameter* CGwmOneDimDistance::makeParameter(initializer_list<DistParamVariant> plist)
{
    if (plist.size() == 2)
    {
        const mat& fp = get<vec>(*(plist.begin()));
        const mat& dp = get<vec>(*(plist.begin() + 1));
        return new OneDimDistanceParameter(fp, dp);
    }
    else return nullptr;
}

vec CGwmOneDimDistance::distance(DistanceParameter* parameter, uword focus)
{
    assert(parameter != nullptr);
    OneDimDistanceParameter* p = (OneDimDistanceParameter*)parameter;
    if (focus < p->total)
    {
        return AbstractDistance(p->focusPoints(focus), p->dataPoints);
    }
    else throw std::runtime_error("Target is out of bounds of data points.");
}
