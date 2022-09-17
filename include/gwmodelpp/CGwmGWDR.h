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
public:
    enum NameFormat
    {
        Fixed,
        VarName,
        PrefixVarName,
        SuffixVariable
    };

    typedef tuple<string, mat, NameFormat> ResultLayerDataItem;

public:
    static GwmRegressionDiagnostic CGwmGWDR::CalcDiagnostic(const mat& x, const vec& y, const mat& betas, const vec& shat);

    static vec Fitted(const mat& x, const mat& betas)
    {
        return sum(betas % x, 1);
    }

    static double RSS(const mat& x, const mat& y, const mat& betas)
    {
        vec r = y - Fitted(x, betas);
        return sum(r % r);
    }

    static double AICc(const mat& x, const mat& y, const mat& betas, const vec& shat)
    {
        double ss = RSS(x, y, betas), n = (double)x.n_rows;
        return n * log(ss / n) + n * log(2 * datum::pi) + n * ((n + shat(0)) / (n - 2 - shat(0)));
    }

public:
    mat betas() const
    {
        return mBetas;
    }

    bool hasHatMatrix() const
    {
        return mHasHatMatrix;
    }

    void setHasHatMatrix(bool flag)
    {
        mHasHatMatrix = flag;
    }

public: // CGwmAlgorithm
    void run();
    bool isValid();

public: // IGwmRegressionAnalysis
    GwmVariable dependentVariable() const
    {
        return mDepVar;
    }

    void setDependentVariable(const GwmVariable& variable)
    {
        mDepVar = variable;
    }

    vector<GwmVariable> independentVariables() const
    {
        return mIndepVars;
    }

    void setIndependentVariables(const vector<GwmVariable>& variables)
    {
        mIndepVars = variables;
    }

    mat regression(const mat& x, const vec& y)
    {
        return regression(x, y);
    }

    mat regressionHatmatrix(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qdiag, mat& S)
    {
        return regressionHatmatrixSerial(x, y, betasSE, shat, qdiag, S);
    }

    GwmRegressionDiagnostic diagnostic() const
    {
        return mDiagnostic;
    }

protected:
    mat regressionSerial(const mat& x, const vec& y);
    mat regressionHatmatrixSerial(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qdiag, mat& S);

protected:
    void createResultLayer(initializer_list<ResultLayerDataItem> items);

private:
    void setXY(mat& x, mat& y, const CGwmSimpleLayer* layer, const GwmVariable& depVar, const vector<GwmVariable>& indepVars);

    bool isStoreS()
    {
        return mHasHatMatrix && (mSourceLayer->featureCount() < 8192);
    }

private:
    vector<CGwmSpatialWeight> mSpatialWeights;
    vector<DistanceParameter*> mDistParameters;

    mat mY;
    mat mX;
    mat mBetas;
    GwmVariable mDepVar;
    vector<GwmVariable> mIndepVars;
    bool mHasHatMatrix;
    GwmRegressionDiagnostic mDiagnostic;
};

#endif  // CGWMGWDR_H