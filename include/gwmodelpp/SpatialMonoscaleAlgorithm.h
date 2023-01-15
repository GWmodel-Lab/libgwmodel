#ifndef SPATIALMONOSCALEALGORITHM_H
#define SPATIALMONOSCALEALGORITHM_H

#include "SpatialAlgorithm.h"
#include "spatialweight/SpatialWeight.h"

namespace gwm
{

/**
 * @brief Interface for spatial algorithm with a single bandwidth. 
 * It defines some interface commonly used in spatial algorithms:
 * 
 * - Getter and setter of spatial weight configuration.
 * 
 * Pointer of this type can be put to these classes:
 * 
 * - GWRBasic
 * - GWSS
 * 
 */
class SpatialMonoscaleAlgorithm : public SpatialAlgorithm
{
public:

    /**
     * @brief Construct a new SpatialMonoscaleAlgorithm object.
     */
    SpatialMonoscaleAlgorithm() {}

    /**
     * @brief Construct a new SpatialMonoscaleAlgorithm object
     * 
     * @param spatialWeight 
     */
    SpatialMonoscaleAlgorithm(const SpatialWeight& spatialWeight, arma::mat coords) : SpatialAlgorithm(coords)
    {
        mSpatialWeight = spatialWeight;
    }

    /**
     * @brief Destroy the SpatialMonoscaleAlgorithm object.
     */
    virtual ~SpatialMonoscaleAlgorithm() {}

public:

    /**
     * @brief Get the spatial weight configuration.
     * 
     * @return Spatial weight configuration object.
     */
    const SpatialWeight& spatialWeight() const
    {
        return mSpatialWeight;
    }

    /**
     * @brief Set the spatial weight configuration.
     * 
     * Use gwmodel_set_gwr_spatial_weight() to set this property to GWRBasic in shared build.
     * 
     * Use gwmodel_set_gwss_spatial_weight() to set this property to GWSS in shared build.
     * 
     * @param spatialWeight Reference of spatial weight configuration object.
     */
    void setSpatialWeight(const SpatialWeight &spatialWeight)
    {
        mSpatialWeight = spatialWeight;
    }

    /**
     * @brief Create a Distance Parameter object. Store in GWSS::mDistanceParameter.
     */
    void createDistanceParameter();

protected:
    SpatialWeight mSpatialWeight;   //!< Spatial weight configuration.    
};

}

#endif  // SPATIALMONOSCALEALGORITHM_H