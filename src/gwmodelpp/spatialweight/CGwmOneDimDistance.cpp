#include "gwmodelpp/spatialweight/CGwmOneDimDistance.h"
#include <assert.h>

#include <exception>


CGwmOneDimDistance::CGwmOneDimDistance() : CGwmDistance()
{

}

CGwmOneDimDistance::CGwmOneDimDistance(const CGwmOneDimDistance &distance) : CGwmDistance(distance)
{
    
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
