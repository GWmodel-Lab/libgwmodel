#ifndef CGWMSPATIALALGORITHM_H
#define CGWMSPATIALALGORITHM_H

#include "CGwmAlgorithm.h"
#include <armadillo>

namespace gwm
{

/**
 * @brief Abstract spatial algorithm class. 
 * This class cannot been constructed.
 * It defines some interface commonly used in spatial algorithms:
 * 
 * - Getter and setter of source layer.
 * - Getter and setter of result layer.
 * - Check if configuration is valid.
 * 
 * Pointer of this type can be put to these classes:
 * 
 * - CGwmGWRBasic
 * - CGwmGWSS
 * 
 */
class CGwmSpatialAlgorithm : public CGwmAlgorithm
{
public:

    /**
     * @brief Construct a new CGwmSpatialAlgorithm object.
     */
    CGwmSpatialAlgorithm() {}

    /**
     * @brief Construct a new CGwmSpatialAlgorithm object.
     * 
     * @param coords Coordinates representing positions of samples.
     */
    CGwmSpatialAlgorithm(const arma::mat& coords) : mCoords(coords) {};

    /**
     * @brief Destroy the CGwmSpatialAlgorithm object.
     */
    virtual ~CGwmSpatialAlgorithm() { mCoords.clear(); }

public:

    /**
     * @brief Get Coords.
     * 
     * @return arma::mat 
     */
    arma::mat coords()
    {
        return mCoords;
    }

    /**
     * @brief Set the Coords object
     * 
     * @param coords Coordinates representing positions of samples.
     */
    void setCoords(const arma::mat& coords)
    {
        mCoords = coords;
    }

    /**
     * @brief Check whether the algorithm's configuration is valid. 
     * 
     * @return true if the algorithm's configuration is valid.
     * @return false if the algorithm's configuration is invalid.
     */
    virtual bool isValid() override;

protected:
    
    arma::mat mCoords;
};

}

#endif  // CGWMSPATIALALGORITHM_H