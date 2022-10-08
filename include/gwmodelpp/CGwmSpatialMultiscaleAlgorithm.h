#ifndef CGWMSPATIALMULTISCALEALGORITHM_H
#define CGWMSPATIALMULTISCALEALGORITHM_H

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
class CGwmSpatialMultiscaleAlgorithm : public CGwmSpatialAlgorithm
{
public:

    /**
     * @brief Construct a new CGwmSpatialMultiscaleAlgorithm object.
     */
    CGwmSpatialMultiscaleAlgorithm();

    /**
     * @brief Destroy the CGwmSpatialMultiscaleAlgorithm object.
     */
    virtual ~CGwmSpatialMultiscaleAlgorithm();

public:

    /**
     * @brief Get the spatial weight configuration.
     * 
     * @return Spatial weight configuration object.
     */
    vector<CGwmSpatialWeight> spatialWeights() const;

    /**
     * @brief Set the spatial weight configuration.
     * 
     * Use gwmodel_set_gwr_spatial_weight() to set this property to CGwmGWRBasic in shared build.
     * 
     * Use gwmodel_set_gwss_spatial_weight() to set this property to CGwmGWSS in shared build.
     * 
     * @param spatialWeights Reference of spatial weight configuration object.
     */
    virtual void setSpatialWeights(const vector<CGwmSpatialWeight> &spatialWeights)
    {
        mSpatialWeights = spatialWeights;
    };
    
    void createDistanceParameter();

protected:
    vector<CGwmSpatialWeight> mSpatialWeights;   //!< Spatial weight configuration.
};

inline vector<CGwmSpatialWeight> CGwmSpatialMultiscaleAlgorithm::spatialWeights() const
{
    return mSpatialWeights;
}

inline void CGwmSpatialMultiscaleAlgorithm::setSpatialWeights(const vector<CGwmSpatialWeight> &spatialWeights)
{
    mSpatialWeights = spatialWeights;
}

#endif  // CGWMSPATIALMultiSCALEALGORITHM_H