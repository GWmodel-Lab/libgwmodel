#include "gwmodelpp/spatialweight/SpatialWeight.h"

using namespace gwm;

SpatialWeight::SpatialWeight()
{

}

SpatialWeight::SpatialWeight(Weight *weight, Distance *distance)
{
    mWeight = weight->clone();
    mDistance = distance->clone();
}

SpatialWeight::SpatialWeight(const SpatialWeight &spatialWeight)
{
    mWeight = spatialWeight.mWeight->clone();
    mDistance = spatialWeight.mDistance->clone();
}

SpatialWeight::~SpatialWeight()
{
    if (mWeight) delete mWeight;
    if (mDistance) delete mDistance;
    mWeight = nullptr;
    mDistance = nullptr;
}

SpatialWeight &SpatialWeight::operator=(const SpatialWeight &&spatialWeight)
{
    if (this == &spatialWeight) return *this;
    if (mWeight) delete mWeight;
    if (mDistance) delete mDistance;
    mWeight = spatialWeight.mWeight->clone();
    mDistance = spatialWeight.mDistance->clone();
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
