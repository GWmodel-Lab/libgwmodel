#ifndef CGWMSPATIALMONOSCALEALGORITHM_H
#define CGWMSPATIALMONOSCALEALGORITHM_H

#include "gwmodelpp.h"
#include "CGwmSpatialAlgorithm.h"
#include "spatialweight/CGwmSpatialWeight.h"

class GWMODELPP_API CGwmSpatialMonoscaleAlgorithm : public CGwmSpatialAlgorithm
{
public:
    CGwmSpatialMonoscaleAlgorithm();
    ~CGwmSpatialMonoscaleAlgorithm();

public:
    CGwmSpatialWeight spatialWeight() const;
    void setSpatialWeight(const CGwmSpatialWeight &spatialWeight);

protected:
    CGwmSpatialWeight mSpatialWeight;
};

inline CGwmSpatialWeight CGwmSpatialMonoscaleAlgorithm::spatialWeight() const
{
    return mSpatialWeight;
}

inline void CGwmSpatialMonoscaleAlgorithm::setSpatialWeight(const CGwmSpatialWeight &spatialWeight)
{
    mSpatialWeight = spatialWeight;
}

#endif  // CGWMSPATIALMONOSCALEALGORITHM_H