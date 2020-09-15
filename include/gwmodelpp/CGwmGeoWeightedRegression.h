#ifndef CGWMGEOWEIGHTEDREGRESSION_H
#define CGWMGEOWEIGHTEDREGRESSION_H

#include "CGwmSpatialMonoscaleAlgorithm.h"
#include "IGwmRegressionAnalysis.h"

class CGwmGeoWeightedRegression : public CGwmSpatialMonoscaleAlgorithm, public IGwmRegressionAnalysis
{
public:
    CGwmGeoWeightedRegression();
    ~CGwmGeoWeightedRegression();

public:
    mat betas() const;

    CGwmSimpleLayer* predictLayer() const;
    void setPredictLayer(CGwmSimpleLayer* layer);

public:     // Implement IGwmRegressionAnalysis
    virtual GwmVariable dependentVariable() const;
    virtual void setDependentVariable(const GwmVariable& variable);

    virtual vector<GwmVariable> independentVariables() const;
    virtual void setIndependentVariables(const vector<GwmVariable>& variables);

    virtual GwmRegressionDiagnostic diagnostic() const;

public:
    virtual bool isValid() override;

    bool hasPredictLayer();

    virtual void initXY(mat& x, mat& y, const GwmVariable& depVar, const vector<GwmVariable>& indepVars);

protected:
    CGwmSimpleLayer* mPredictLayer;

    GwmVariable mDepVar;
    vector<GwmVariable> mIndepVars;

    mat mX;
    vec mY;
    mat mBetas;

    GwmRegressionDiagnostic mDiagnostic;
};

inline CGwmSimpleLayer* CGwmGeoWeightedRegression::predictLayer() const
{
    return mPredictLayer;
}

inline void CGwmGeoWeightedRegression::setPredictLayer(CGwmSimpleLayer* layer)
{
    mPredictLayer = layer;
}

inline GwmVariable CGwmGeoWeightedRegression::dependentVariable() const
{
    return mDepVar;
}

inline void CGwmGeoWeightedRegression::setDependentVariable(const GwmVariable& variable)
{
    mDepVar = variable;
}

inline vector<GwmVariable> CGwmGeoWeightedRegression::independentVariables() const
{
    return mIndepVars;
}

inline void CGwmGeoWeightedRegression::setIndependentVariables(const vector<GwmVariable>& variables)
{
    mIndepVars = variables;
}

inline GwmRegressionDiagnostic CGwmGeoWeightedRegression::diagnostic() const
{
    return mDiagnostic;
}

inline bool CGwmGeoWeightedRegression::hasPredictLayer()
{
    return mPredictLayer != nullptr;
}

#endif  // CGWMGEOWEIGHTEDREGRESSION_H