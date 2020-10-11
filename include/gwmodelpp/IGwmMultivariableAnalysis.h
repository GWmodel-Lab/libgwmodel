#ifndef IGWMMULTIVARIABLEANALYSIS_H
#define IGWMMULTIVARIABLEANALYSIS_H

#include <vector>
#include "GwmVariable.h"

using namespace std;

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
    virtual vector<GwmVariable> variables() const = 0;

    /**
     * @brief Set variables.
     * 
     * @param variables Vector of variables.
     */
    virtual void setVariables(const vector<GwmVariable>& variables) = 0;
};


#endif  // IGWMMULTIVARIABLEANALYSIS_H