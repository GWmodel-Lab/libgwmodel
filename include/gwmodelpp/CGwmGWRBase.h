#ifndef CGWMGWRBASE_H
#define CGWMGWRBASE_H

#include "CGwmSpatialMonoscaleAlgorithm.h"
#include "IGwmRegressionAnalysis.h"

class CGwmGWRBase : public CGwmSpatialMonoscaleAlgorithm, public IGwmRegressionAnalysis
{
public:
    CGwmGWRBase();
    ~CGwmGWRBase();

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

inline CGwmSimpleLayer* CGwmGWRBase::predictLayer() const
{
    return mPredictLayer;
}

inline void CGwmGWRBase::setPredictLayer(CGwmSimpleLayer* layer)
{
    mPredictLayer = layer;
}

inline GwmVariable CGwmGWRBase::dependentVariable() const
{
    return mDepVar;
}

inline void CGwmGWRBase::setDependentVariable(const GwmVariable& variable)
{
    mDepVar = variable;
}

inline vector<GwmVariable> CGwmGWRBase::independentVariables() const
{
    return mIndepVars;
}

inline void CGwmGWRBase::setIndependentVariables(const vector<GwmVariable>& variables)
{
    mIndepVars = variables;
}

inline GwmRegressionDiagnostic CGwmGWRBase::diagnostic() const
{
    return mDiagnostic;
}

inline bool CGwmGWRBase::hasPredictLayer()
{
    return mPredictLayer != nullptr;
}

#endif  // CGWMGWRBASE_H