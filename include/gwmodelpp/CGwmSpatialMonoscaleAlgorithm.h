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
    CGwmSpatialMonoscaleAlgorithm() {}

    /**
     * @brief Construct a new CGwmSpatialMonoscaleAlgorithm object
     * 
     * @param spatialWeight 
     */
    CGwmSpatialMonoscaleAlgorithm(const CGwmSpatialWeight& spatialWeight, arma::mat coords) : CGwmSpatialAlgorithm(coords)
    {
        mSpatialWeight = spatialWeight;
    }

    /**
     * @brief Destroy the CGwmSpatialMonoscaleAlgorithm object.
     */
    virtual ~CGwmSpatialMonoscaleAlgorithm() {}

public:

    /**
     * @brief Get the spatial weight configuration.
     * 
     * @return Spatial weight configuration object.
     */
    CGwmSpatialWeight spatialWeight() const
    {
        return mSpatialWeight;
    }

    /**
     * @brief Set the spatial weight configuration.
     * 
     * Use gwmodel_set_gwr_spatial_weight() to set this property to CGwmGWRBasic in shared build.
     * 
     * Use gwmodel_set_gwss_spatial_weight() to set this property to CGwmGWSS in shared build.
     * 
     * @param spatialWeight Reference of spatial weight configuration object.
     */
    void setSpatialWeight(const CGwmSpatialWeight &spatialWeight)
    {
        mSpatialWeight = spatialWeight;
    }

    /**
     * @brief Create a Distance Parameter object. Store in CGwmGWSS::mDistanceParameter.
     */
    void createDistanceParameter();

protected:
    CGwmSpatialWeight mSpatialWeight;   //!< Spatial weight configuration.    
};

#endif  // CGWMSPATIALMONOSCALEALGORITHM_H