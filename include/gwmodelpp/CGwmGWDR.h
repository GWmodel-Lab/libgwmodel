#ifndef CGWMGWDR_H
#define CGWMGWDR_H

#include <vector>
#include <armadillo>
#include "CGwmSpatialAlgorithm.h"
#include "spatialweight/CGwmSpatialWeight.h"
#include "IGwmRegressionAnalysis.h"

using namespace std;
using namespace arma;

class CGwmGWDR : CGwmSpatialAlgorithm, IGwmRegressionAnalysis
{

public: // CGwmAlgorithm
    void run();
    bool isValid();

public: // IGwmRegressionAnalysis
    GwmVariable dependentVariable() const;
    void setDependentVariable(const GwmVariable& variable);

    vector<GwmVariable> independentVariables() const;
    void setIndependentVariables(const vector<GwmVariable>& variables);

    mat regression(const mat& x, const vec& y);
    mat regressionHatmatrix(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qdiag, mat& S);

    GwmRegressionDiagnostic diagnostic() const;

private:
    vector<CGwmSpatialWeight> mSpatialWeights;

    mat mY;
    mat mX;
};

#endif  // CGWMGWDR_H