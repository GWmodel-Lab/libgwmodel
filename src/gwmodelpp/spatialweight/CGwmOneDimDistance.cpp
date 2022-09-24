#include "gwmodelpp/spatialweight/CGwmOneDimDistance.h"
#include <assert.h>

#include <exception>


CGwmOneDimDistance::CGwmOneDimDistance() : CGwmDistance()
{

}

CGwmOneDimDistance::CGwmOneDimDistance(const CGwmOneDimDistance &distance) : CGwmDistance(distance)
{
    
}

CGwmDistance::Parameter* CGwmOneDimDistance::makeParameter(initializer_list<DistParamVariant> plist)
{
    if (mParameter != nullptr) delete mParameter;
    if (plist.size() == 2)
    {
        const mat& fp = get<vec>(*(plist.begin()));
        const mat& dp = get<vec>(*(plist.begin() + 1));
        mParameter = new Parameter(fp, dp);
    }
    else mParameter = nullptr;
    return mParameter;
}

vec CGwmOneDimDistance::distance(uword focus)
{
    assert(mParameter != nullptr);
    Parameter* p = static_cast<Parameter*>(mParameter);
    if (focus < p->total)
    {
        return AbstractDistance(p->focusPoints(focus), p->dataPoints);
    }
    else throw std::runtime_error("Target is out of bounds of data points.");
}
