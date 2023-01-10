#ifndef SPATIALALGORITHM_H
#define SPATIALALGORITHM_H

#include "Algorithm.h"
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
 * - GWRBasic
 * - GWSS
 * 
 */
class SpatialAlgorithm : public Algorithm
{
public:

    /**
     * @brief Construct a new SpatialAlgorithm object.
     */
    SpatialAlgorithm() {}

    /**
     * @brief Construct a new SpatialAlgorithm object.
     * 
     * @param coords Coordinates representing positions of samples.
     */
    SpatialAlgorithm(const arma::mat& coords) : mCoords(coords) {};

    /**
     * @brief Destroy the SpatialAlgorithm object.
     */
    virtual ~SpatialAlgorithm() { mCoords.clear(); }

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

#endif  // SPATIALALGORITHM_H