#include "gwmodelpp/spatialweight/OneDimDistance.h"
#include <assert.h>

#include <exception>

using namespace std;
using namespace arma;
using namespace gwm;


OneDimDistance::OneDimDistance() : Distance()
{

}

OneDimDistance::OneDimDistance(const OneDimDistance &distance) : Distance(distance)
{
    
}

void OneDimDistance::makeParameter(initializer_list<DistParamVariant> plist)
{
    if (plist.size() == 2)
    {
        const mat& fp = get<vec>(*(plist.begin()));
        const mat& dp = get<vec>(*(plist.begin() + 1));
        mParameter = make_unique<Parameter>(fp, dp);
    }
    else
    {
        mParameter.reset(nullptr);
        throw std::runtime_error("The number of parameters must be 2.");
    }
}

vec OneDimDistance::distance(uword focus)
{
    if (mParameter == nullptr) throw std::runtime_error("Parameter is nullptr.");

    if (focus < mParameter->total)
    {
        return AbstractDistance(mParameter->focusPoints(focus), mParameter->dataPoints);
    }
    else throw std::runtime_error("Target is out of bounds of data points.");
}

vec OneDimDistance::noAbsdistance(uword focus)
{
    if (mParameter == nullptr) throw std::runtime_error("Parameter is nullptr.");

    if (focus < mParameter->total)
    {
        const double& out_loc=mParameter->focusPoints(focus);
        const arma::vec& in_locs=mParameter->dataPoints;
        return out_loc - in_locs;
    }
    else throw std::runtime_error("Target is out of bounds of data points.");
}


double OneDimDistance::maxDistance()
{
    if (mParameter == nullptr) throw std::runtime_error("Parameter is nullptr.");
    double maxD = 0.0;
    for (uword i = 0; i < mParameter->total; i++)
    {
        double d = max(AbstractDistance(mParameter->focusPoints(i), mParameter->dataPoints));
        maxD = d > maxD ? d : maxD;
    }
    return maxD;
}

double OneDimDistance::minDistance()
{
    if (mParameter == nullptr) throw std::runtime_error("Parameter is nullptr.");
    double minD = DBL_MAX;
    for (uword i = 0; i < mParameter->total; i++)
    {
        double d = min(AbstractDistance(mParameter->focusPoints(i), mParameter->dataPoints));
        minD = d < minD ? d : minD;
    }
    return minD;
}
