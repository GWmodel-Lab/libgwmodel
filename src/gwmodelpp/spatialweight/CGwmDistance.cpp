#include "gwmodelpp/spatialweight/CGwmDistance.h"
#include <assert.h>

unordered_map<CGwmDistance::DistanceType, string> CGwmDistance::TypeNameMapper =
{
    std::make_pair(CGwmDistance::DistanceType::CRSDistance, "CRSDistance"),
    std::make_pair(CGwmDistance::DistanceType::MinkwoskiDistance, "MinkwoskiDistance"),
    std::make_pair(CGwmDistance::DistanceType::DMatDistance, "DMatDistance")
};

double CGwmDistance::maxDistance(uword total, DistanceParameter* parameter)
{
    assert(parameter != nullptr);
    double maxD = 0.0;
    for (uword i = 0; i < total; i++)
    {
        double d = max(distance(parameter, i));
        maxD = d > maxD ? d : maxD;
    }
    return maxD;
}

double CGwmDistance::minDistance(uword total, DistanceParameter* parameter)
{
    assert(parameter != nullptr);
    double minD = DBL_MAX;
    for (uword i = 0; i < total; i++)
    {
        double d = min(distance(parameter, i));
        minD = d < minD ? d : minD;
    }
    return minD;
}
