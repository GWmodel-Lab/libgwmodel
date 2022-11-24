#ifndef CGWMGWRBASE_H
#define CGWMGWRBASE_H

#include "CGwmSpatialMonoscaleAlgorithm.h"
#include "IGwmRegressionAnalysis.h"

class CGwmGWRBase : public CGwmSpatialMonoscaleAlgorithm, public IGwmRegressionAnalysis
{
public:
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
    CGwmGWRBase() {}

    CGwmGWRBase(const arma::mat& x, const arma::vec& y, const CGwmSpatialWeight& spatialWeight, const arma::mat& coords) : CGwmSpatialMonoscaleAlgorithm(spatialWeight, coords)
    {
        mX = x;
        mY = y;
    }

    ~CGwmGWRBase() {}

public:
    arma::mat betas() const { return mBetas; }

public:     // Implement IGwmRegressionAnalysis
    virtual arma::vec dependentVariable() const { return mY; }
    virtual void setDependentVariable(const arma::vec& y) { mY = y; }

    virtual arma::mat independentVariables() const { return mX; }
    virtual void setIndependentVariables(const arma::mat& x) { mX = x; }

    virtual GwmRegressionDiagnostic diagnostic() const { return mDiagnostic; }

public:
    virtual bool isValid() override;

protected:

    arma::mat mX;
    arma::vec mY;
    arma::mat mBetas;

    GwmRegressionDiagnostic mDiagnostic;
};

#endif  // CGWMGWRBASE_H