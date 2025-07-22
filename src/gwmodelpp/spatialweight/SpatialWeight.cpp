#include "gwmodelpp/spatialweight/SpatialWeight.h"

using namespace gwm;

SpatialWeight &SpatialWeight::operator=(SpatialWeight &&spatialWeight)
{
    if (this == &spatialWeight) return *this;
    mWeight = std::move(spatialWeight.mWeight);
    mDistance = std::move(spatialWeight.mDistance);
    return *this;
}

SpatialWeight &SpatialWeight::operator=(const SpatialWeight &spatialWeight)
{
    if (this == &spatialWeight) return *this;
    mWeight = std::move(spatialWeight.mWeight->clone());
    mDistance = std::move(spatialWeight.mDistance->clone());
    return *this;
}

bool SpatialWeight::isValid()
{
    return !((mWeight == 0) || (mDistance == 0));
}
