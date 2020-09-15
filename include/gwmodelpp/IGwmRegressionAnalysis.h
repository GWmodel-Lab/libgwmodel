#ifndef IGWMREGRESSIONANALYSIS_H
#define IGWMREGRESSIONANALYSIS_H

#include <vector>
#include "GwmVariable.h"
#include "GwmRegressionDiagnostic.h"

using namespace std;

class IGwmRegressionAnalysis
{
    virtual GwmVariable dependentVariable() const = 0;
    virtual void setDependentVariable(const GwmVariable& variable) = 0;

    virtual vector<GwmVariable> independentVariables() const = 0;
    virtual void setIndependentVariables(const vector<GwmVariable>& variables) = 0;

    virtual mat regression(const mat& x, const vec& y) = 0;
    virtual mat regressionHatmatrix(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qdiag, mat& S) = 0;

    virtual GwmRegressionDiagnostic diagnostic() const = 0;
};

#endif  // IGWMREGRESSIONANALYSIS_H