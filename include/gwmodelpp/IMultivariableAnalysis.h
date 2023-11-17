#ifndef IMULTIVARIABLEANALYSIS_H
#define IMULTIVARIABLEANALYSIS_H

#include <armadillo>

namespace gwm
{

/**
 * @brief Interface for multivariable analysis. 
 * It defines some interface commonly used in spatial algorithms:
 * 
 * - Getter and setter of variables.
 * 
 * Pointer of this type can be put to these classes:
 * 
 * - GWSS
 * 
 */
struct IMultivariableAnalysis
{
    /**
     * @brief Get variables. 
     * 
     * @return Vector of variables.
     */
    virtual const arma::mat& variables() const = 0;

    /**
     * @brief Set variables.
     * 
     * @param x Vector of variables.
     */
    virtual void setVariables(const arma::mat& x) = 0;

    /**
     * @brief Run analysis algorithm.
     * 
     */
    virtual void run() = 0;
};

}


#endif  // IMULTIVARIABLEANALYSIS_H