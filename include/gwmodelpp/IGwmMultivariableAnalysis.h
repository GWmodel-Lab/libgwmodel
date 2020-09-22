#ifndef IGWMMULTIVARIABLEANALYSIS_H
#define IGWMMULTIVARIABLEANALYSIS_H

#include <vector>
#include "GwmVariable.h"

using namespace std;

struct IGwmMultivariableAnalysis
{
    virtual vector<GwmVariable> variables() const = 0;
    virtual void setVariables(const vector<GwmVariable>& variables) = 0;
};


#endif  // IGWMMULTIVARIABLEANALYSIS_H