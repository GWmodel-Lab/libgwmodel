#ifndef CGWMROBUSTGWR_H
#define CGWMROBUSTGWR_H

#include <utility>
#include <string>
#include <initializer_list>
#include "CGwmGWRBase.h"
#include "GwmRegressionDiagnostic.h"
#include "IGwmBandwidthSelectable.h"
#include "IGwmVarialbeSelectable.h"
#include "IGwmParallelizable.h"
#include "CGwmGWRBasic.h"

using namespace arma;
class CGwmRobustGWR : public CGwmGWRBasic
{
private:
    typedef mat (CGwmRobustGWR::*RegressionHatmatrix)(const mat &, const vec &, mat &, vec &, vec &, mat &);

    static GwmRegressionDiagnostic CalcDiagnostic(const mat &x, const vec &y, const mat &betas, const vec &shat);

public:
    CGwmRobustGWR();
    ~CGwmRobustGWR();

public:
    bool filtered() const;
    void setFiltered(bool value);

public: // Implement IGwmRegressionAnalysis
    //mat regression(const mat &x, const vec &y) override;
    mat predict(const mat& locations) override;
    mat fit() override;
    mat regressionHatmatrix(const mat &x, const vec &y, mat &betasSE, vec &shat, vec &qdiag, mat &S);

private:
    //mat regressionHatmatrixSerial(const mat &x, const vec &y, mat &betasSE, vec &shat, vec &qDiag, mat &S);
    //mat predictSerial(const mat& locations, const mat& x, const vec& y);
    mat fitSerial(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qDiag, mat& S);
       
#ifdef ENABLE_OPENMP
    //mat predictOmp(const mat& locations, const mat& x, const vec& y);
    mat fitOmp(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qDiag, mat& S);
#endif

public: // Implement CGwmAlgorithm
    //void run();

protected:
    mat robustGWRCaliFirst(const mat &x, const vec &y, mat &betasSE, vec &shat, vec &qDiag, mat &S);
    // 第二种解法
    mat robustGWRCaliSecond(const mat &x, const vec &y, mat &betasSE, vec &shat, vec &qDiag, mat &S);
    // 计算二次权重函数
    vec filtWeight(vec residual, double mse);

public : // Implement IGwmParallelizable
    void setParallelType(const ParallelType &type) override;

protected:
    void createPredictionDistanceParameter(const arma::mat& locations);

private:
    bool mFiltered;

    mat mS;
    vec mWeightMask;
    
    RegressionHatmatrix mfitFunction = &CGwmRobustGWR::fitSerial;
    
};

inline bool CGwmRobustGWR::filtered() const
{
    return mFiltered;
}

inline void CGwmRobustGWR::setFiltered(bool value)
{
    if (value)
    {
        this->mFiltered = true;
    }
    else
    {
        this->mFiltered = false;
    }
}

#endif // CGWMROBUSTGWR_H