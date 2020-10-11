#ifndef CGWMSPATIALMONOSCALEALGORITHM_H
#define CGWMSPATIALMONOSCALEALGORITHM_H

#include "CGwmSpatialAlgorithm.h"
#include "spatialweight/CGwmSpatialWeight.h"

/**
 * @brief Interface for spatial algorithm with a single bandwidth. 
 * It defines some interface commonly used in spatial algorithms:
 * 
 * - Getter and setter of spatial weight configuration.
 * 
 * Pointer of this type can be put to these classes:
 * 
 * - CGwmGWRBasic
 * - CGwmGWSS
 * 
 */
class CGwmSpatialMonoscaleAlgorithm : public CGwmSpatialAlgorithm
{
public:

    /**
     * @brief Construct a new CGwmSpatialMonoscaleAlgorithm object.
     */
    CGwmSpatialMonoscaleAlgorithm();

    /**
     * @brief Destroy the CGwmSpatialMonoscaleAlgorithm object.
     */
    ~CGwmSpatialMonoscaleAlgorithm();

public:

    /**
     * @brief Get the spatial weight configuration.
     * 
     * @return Spatial weight configuration object.
     */
    CGwmSpatialWeight spatialWeight() const;

    /**
     * @brief Set the spatial weight configuration.
     * 
     * @param spatialWeight Reference of spatial weight configuration object.
     */
    void setSpatialWeight(const CGwmSpatialWeight &spatialWeight);

protected:
    CGwmSpatialWeight mSpatialWeight;   //!< Spatial weight configuration.
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