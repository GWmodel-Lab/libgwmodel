#ifndef IMULTIVARIABLEANALYSIS_H
#define IMULTIVARIABLEANALYSIS_H

#include "armadillo_config.h"

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

struct IMultivariateAnalysis
{
    /**
     * @brief Get response variables. 
     * 
     * @return arma::mat of variables.
     */
    virtual const arma::mat& variables1() const = 0;

    /**
     * @brief Set variables.
     * 
     * @param y arma::mat of variables.
     */
    virtual void setVariables1(const arma::mat& y) = 0;

    /**
     * @brief Get variables. 
     * 
     * @return arma::mat of variables.
     */
    virtual const arma::mat& variables2() const = 0;

    /**
     * @brief Set variables.
     * 
     * @param x arma::mat of variables.
     */
    virtual void setVariables2(const arma::mat& x) = 0;


    /**
     * @brief Run analysis algorithm.
     * 
     */
    virtual void run() = 0;
};

}


#endif  // IMULTIVARIABLEANALYSIS_H