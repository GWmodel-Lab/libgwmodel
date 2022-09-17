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

using namespace std;
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
    mat regression(const mat &x, const vec &y) override;
    mat regressionHatmatrix(const mat &x, const vec &y, mat &betasSE, vec &shat, vec &qdiag, mat &S) override;

private:
    mat regressionHatmatrixSerial(const mat &x, const vec &y, mat &betasSE, vec &shat, vec &qDiag, mat &S);

public: // Implement CGwmAlgorithm
    void run() override;

protected:
    mat robustGWRCaliFirst(const mat &x, const vec &y, mat &betasSE, vec &shat, vec &qDiag, mat &S);
    // 第二种解法
    mat robustGWRCaliSecond(const mat &x, const vec &y, mat &betasSE, vec &shat, vec &qDiag, mat &S);
    // 计算二次权重函数
    vec filtWeight(vec residual, double mse);
#ifdef ENABLE_OpenMP
    mat regressionHatmatrixOmp(const mat &x, const vec &y, mat &betasSE, vec &shat, vec &qDiag, mat &S); override
#endif
public : // Implement IGwmParallelizable
    void setParallelType(const ParallelType &type);

private:
    bool mFiltered;

    mat mS;
    vec mWeightMask;

    RegressionHatmatrix mRegressionHatmatrixFunction = &CGwmRobustGWR::regressionHatmatrixSerial;
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