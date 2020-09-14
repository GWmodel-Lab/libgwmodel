#ifndef CGWMSPATIALMONOSCALEALGORITHM_H
#define CGWMSPATIALMONOSCALEALGORITHM_H

#include "CGwmSpatialAlgorithm.h"
#include "spatialweight/CGwmSpatialWeight.h"

class CGwmSpatialMonoscaleAlgorithm : public CGwmSpatialAlgorithm
{
public:
    CGwmSpatialMonoscaleAlgorithm();
    ~CGwmSpatialMonoscaleAlgorithm();

public:
    CGwmSpatialWeight spatialWeight() const;
    void setSpatialWeight(const CGwmSpatialWeight &spatialWeight);

private:
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