#ifndef CGWMSPATIALWEIGHT_H
#define CGWMSPATIALWEIGHT_H

#include "spatialweight/CGwmWeight.h"
#include "spatialweight/CGwmDistance.h"

#include "spatialweight/CGwmBandwidthWeight.h"

#include "spatialweight/CGwmCRSDistance.h"
#include "spatialweight/CGwmMinkwoskiDistance.h"
#include "spatialweight/CGwmDMatDistance.h"

class CGwmSpatialWeight
{
public:
    CGwmSpatialWeight();
    CGwmSpatialWeight(CGwmWeight* weight, CGwmDistance* distance);
    CGwmSpatialWeight(const CGwmSpatialWeight& spatialWeight);
    ~CGwmSpatialWeight();

    CGwmWeight *weight() const;
    void setWeight(CGwmWeight *weight);
    void setWeight(CGwmWeight& weight);
    void setWeight(CGwmWeight&& weight);

    template<typename T>
    T* weight() const { return nullptr; }

    CGwmDistance *distance() const;
    void setDistance(CGwmDistance *distance);
    void setDistance(CGwmDistance& distance);
    void setDistance(CGwmDistance&& distance);

    template<typename T>
    T* distance() const { return nullptr; }

public:
    CGwmSpatialWeight& operator=(const CGwmSpatialWeight& spatialWeight);
    CGwmSpatialWeight& operator=(const CGwmSpatialWeight&& spatialWeight);

public:
    virtual vec weightVector(DistanceParameter* parameter, uword focus);
    virtual bool isValid();

private:
    CGwmWeight* mWeight = nullptr;
    CGwmDistance* mDistance = nullptr;
};

inline CGwmWeight *CGwmSpatialWeight::weight() const
{
    return mWeight;
}

template<>
inline CGwmBandwidthWeight* CGwmSpatialWeight::weight<CGwmBandwidthWeight>() const
{
    return static_cast<CGwmBandwidthWeight*>(mWeight);
}

inline void CGwmSpatialWeight::setWeight(CGwmWeight *weight)
{
    if (mWeight) delete mWeight;
    mWeight = weight;
}

inline void CGwmSpatialWeight::setWeight(CGwmWeight& weight)
{
    if (mWeight) delete mWeight;
    mWeight = weight.clone();
}

inline void CGwmSpatialWeight::setWeight(CGwmWeight&& weight)
{
    if (mWeight) delete mWeight;
    mWeight = weight.clone();
}

inline CGwmDistance *CGwmSpatialWeight::distance() const
{
    return mDistance;
}

template<>
inline CGwmCRSDistance* CGwmSpatialWeight::distance<CGwmCRSDistance>() const
{
    return static_cast<CGwmCRSDistance*>(mDistance);
}

template<>
inline CGwmMinkwoskiDistance* CGwmSpatialWeight::distance<CGwmMinkwoskiDistance>() const
{
    return static_cast<CGwmMinkwoskiDistance*>(mDistance);
}

template<>
inline CGwmDMatDistance* CGwmSpatialWeight::distance<CGwmDMatDistance>() const
{
    return static_cast<CGwmDMatDistance*>(mDistance);
}

inline void CGwmSpatialWeight::setDistance(CGwmDistance *distance)
{
    if (mDistance) delete mDistance;
    mDistance = distance;
}

inline void CGwmSpatialWeight::setDistance(CGwmDistance& distance)
{
    if (mDistance) delete mDistance;
    mDistance = distance.clone();
}

inline void CGwmSpatialWeight::setDistance(CGwmDistance&& distance)
{
    if (mDistance) delete mDistance;
    mDistance = distance.clone();
}

inline vec CGwmSpatialWeight::weightVector(DistanceParameter* parameter, uword focus)
{
    return mWeight->weight(mDistance->distance(parameter, focus));
}

#endif // CGWMSPATIALWEIGHT_H
