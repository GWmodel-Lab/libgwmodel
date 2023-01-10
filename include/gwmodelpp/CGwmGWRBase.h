#ifndef CGWMGWRBASE_H
#define CGWMGWRBASE_H

#include "CGwmSpatialMonoscaleAlgorithm.h"
#include "IGwmRegressionAnalysis.h"

namespace gwm
{

class CGwmGWRBase : public CGwmSpatialMonoscaleAlgorithm, public IGwmRegressionAnalysis
{
public:
    static arma::vec Fitted(const arma::mat& x, const arma::mat& betas)
    {
        return sum(betas % x, 1);
    }

    static double RSS(const arma::mat& x, const arma::mat& y, const arma::mat& betas)
    {
        arma::vec r = y - Fitted(x, betas);
        return sum(r % r);
    }

    static double AICc(const arma::mat& x, const arma::mat& y, const arma::mat& betas, const arma::vec& shat)
    {
        double ss = RSS(x, y, betas), n = (double)x.n_rows;
        return n * log(ss / n) + n * log(2 * arma::datum::pi) + n * ((n + shat(0)) / (n - 2 - shat(0)));
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
    virtual arma::vec dependentVariable() const override { return mY; }
    virtual void setDependentVariable(const arma::vec& y) override { mY = y; }

    virtual arma::mat independentVariables() const override { return mX; }
    virtual void setIndependentVariables(const arma::mat& x) override { mX = x; }

    virtual bool hasIntercept() const override { return mHasIntercept; }
    virtual void setHasIntercept(const bool has) override { mHasIntercept = has; }

    virtual GwmRegressionDiagnostic diagnostic() const override { return mDiagnostic; }

public:
    virtual bool isValid() override;

protected:

    arma::mat mX;
    arma::vec mY;
    arma::mat mBetas;
    bool mHasIntercept = true;

    GwmRegressionDiagnostic mDiagnostic;
};

}

#endif  // CGWMGWRBASE_H