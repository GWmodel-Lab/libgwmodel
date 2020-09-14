#include "spatialweight/CGwmDistance.h"

unordered_map<CGwmDistance::DistanceType, string> CGwmDistance::TypeNameMapper =
{
    std::make_pair(CGwmDistance::DistanceType::CRSDistance, "CRSDistance"),
    std::make_pair(CGwmDistance::DistanceType::MinkwoskiDistance, "MinkwoskiDistance"),
    std::make_pair(CGwmDistance::DistanceType::DMatDistance, "DMatDistance")
};

double CGwmDistance::maxDistance()
{
    double maxD = 0.0;
    for (int i = 0; i < mTotal; i++)
    {
        double d = max(distance(i));
        maxD = d > maxD ? d : maxD;
    }
    return maxD;
}

double CGwmDistance::minDistance()
{
    double minD = DBL_MAX;
    for (int i = 0; i < mTotal; i++)
    {
        double d = min(distance(i));
        minD = d < minD ? d : minD;
    }
    return minD;
}
