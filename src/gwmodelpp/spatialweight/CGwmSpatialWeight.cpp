#include "spatialweight/CGwmSpatialWeight.h"

CGwmSpatialWeight::CGwmSpatialWeight()
{

}

CGwmSpatialWeight::CGwmSpatialWeight(CGwmWeight *weight, CGwmDistance *distance)
{
    mWeight = weight->clone();
    mDistance = distance->clone();
}

CGwmSpatialWeight::CGwmSpatialWeight(const CGwmSpatialWeight &spatialWeight)
{
    mWeight = spatialWeight.mWeight->clone();
    mDistance = spatialWeight.mDistance->clone();
}

CGwmSpatialWeight::~CGwmSpatialWeight()
{
    if (mWeight) delete mWeight;
    if (mDistance) delete mDistance;
    mWeight = nullptr;
    mDistance = nullptr;
}

CGwmSpatialWeight &CGwmSpatialWeight::operator=(const CGwmSpatialWeight &&spatialWeight)
{
    if (this == &spatialWeight) return *this;
    if (mWeight) delete mWeight;
    if (mDistance) delete mDistance;
    mWeight = spatialWeight.mWeight->clone();
    mDistance = spatialWeight.mDistance->clone();
    return *this;
}

CGwmSpatialWeight &CGwmSpatialWeight::operator=(const CGwmSpatialWeight &spatialWeight)
{
    if (this == &spatialWeight) return *this;
    if (mWeight) delete mWeight;
    if (mDistance) delete mDistance;
    mWeight = spatialWeight.mWeight->clone();
    mDistance = spatialWeight.mDistance->clone();
    return *this;
}

bool CGwmSpatialWeight::isValid()
{
    return !((mWeight == 0) || (mDistance == 0));
}
