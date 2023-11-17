#include "gwmodelpp/spatialweight/SpatialWeight.h"

using namespace gwm;


SpatialWeight::~SpatialWeight()
{
    if (mWeight) delete mWeight;
    if (mDistance) delete mDistance;
    mWeight = nullptr;
    mDistance = nullptr;
}

SpatialWeight &SpatialWeight::operator=(SpatialWeight &&spatialWeight)
{
    if (this == &spatialWeight) return *this;
    if (mWeight) delete mWeight;
    if (mDistance) delete mDistance;
    mWeight = spatialWeight.mWeight;
    mDistance = spatialWeight.mDistance;
    spatialWeight.mWeight = nullptr;
    spatialWeight.mDistance = nullptr;
    return *this;
}

SpatialWeight &SpatialWeight::operator=(const SpatialWeight &spatialWeight)
{
    if (this == &spatialWeight) return *this;
    if (mWeight) delete mWeight;
    if (mDistance) delete mDistance;
    mWeight = spatialWeight.mWeight->clone();
    mDistance = spatialWeight.mDistance->clone();
    return *this;
}

bool SpatialWeight::isValid()
{
    return !((mWeight == 0) || (mDistance == 0));
}
