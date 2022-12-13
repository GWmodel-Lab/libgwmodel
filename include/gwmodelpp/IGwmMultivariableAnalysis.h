#ifndef IGWMMULTIVARIABLEANALYSIS_H
#define IGWMMULTIVARIABLEANALYSIS_H

#include <armadillo>

/**
 * @interface IGwmMultivariableAnalysis
 * @brief Interface for multivariable analysis. 
 * It defines some interface commonly used in spatial algorithms:
 * 
 * - Getter and setter of variables.
 * 
 * Pointer of this type can be put to these classes:
 * 
 * - CGwmGWSS
 * 
 */
struct IGwmMultivariableAnalysis
{
    /**
     * @brief Get variables. 
     * 
     * @return Vector of variables.
     */
    virtual mat variables() const = 0;

    /**
     * @brief Set variables.
     * 
     * @param variables Vector of variables.
     */
    virtual void setVariables(const mat& x) = 0;

    /**
     * @brief Run analysis algorithm.
     * 
     */
    virtual void run() = 0;
};


#endif  // IGWMMULTIVARIABLEANALYSIS_H