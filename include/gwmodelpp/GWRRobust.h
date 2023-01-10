#ifndef GWRROBUST_H
#define GWRROBUST_H

#include <utility>
#include <string>
#include <initializer_list>
#include "GWRBase.h"
#include "RegressionDiagnostic.h"
#include "IBandwidthSelectable.h"
#include "IVarialbeSelectable.h"
#include "IParallelizable.h"
#include "GWRBasic.h"

namespace gwm
{

class GWRRobust : public GWRBasic
{
private:
    typedef arma::mat (GWRRobust::*RegressionHatmatrix)(const arma::mat &, const arma::vec &, arma::mat &, arma::vec &, arma::vec &, arma::mat &);

    static RegressionDiagnostic CalcDiagnostic(const arma::mat &x, const arma::vec &y, const arma::mat &betas, const arma::vec &shat);

public:
    GWRRobust();
    ~GWRRobust();

public:
    bool filtered() const;
    void setFiltered(bool value);

public: // Implement IRegressionAnalysis
    //arma::mat regression(const arma::mat &x, const arma::vec &y) override;
    arma::mat predict(const arma::mat& locations) override;
    arma::mat fit() override;
    arma::mat regressionHatmatrix(const arma::mat &x, const arma::vec &y, arma::mat &betasSE, arma::vec &shat, arma::vec &qdiag, arma::mat &S);

private:
    //arma::mat regressionHatmatrixSerial(const arma::mat &x, const arma::vec &y, arma::mat &betasSE, arma::vec &shat, arma::vec &qDiag, arma::mat &S);
    //arma::mat predictSerial(const arma::mat& locations, const arma::mat& x, const arma::vec& y);
    arma::mat fitSerial(const arma::mat& x, const arma::vec& y, arma::mat& betasSE, arma::vec& shat, arma::vec& qDiag, arma::mat& S);
       
#ifdef ENABLE_OPENMP
    //arma::mat predictOmp(const arma::mat& locations, const arma::mat& x, const arma::vec& y);
    arma::mat fitOmp(const arma::mat& x, const arma::vec& y, arma::mat& betasSE, arma::vec& shat, arma::vec& qDiag, arma::mat& S);
#endif

public: // Implement Algorithm
    //void run();

protected:
    arma::mat robustGWRCaliFirst(const arma::mat &x, const arma::vec &y, arma::mat &betasSE, arma::vec &shat, arma::vec &qDiag, arma::mat &S);
    // 第二种解法
    arma::mat robustGWRCaliSecond(const arma::mat &x, const arma::vec &y, arma::mat &betasSE, arma::vec &shat, arma::vec &qDiag, arma::mat &S);
    // 计算二次权重函数
    arma::vec filtWeight(arma::vec residual, double mse);

public : // Implement IParallelizable
    void setParallelType(const ParallelType &type) override;

protected:
    void createPredictionDistanceParameter(const arma::mat& locations);

private:
    bool mFiltered;

    arma::mat mS;
    arma::vec mWeightMask;
    
    RegressionHatmatrix mfitFunction = &GWRRobust::fitSerial;
    
};

inline bool GWRRobust::filtered() const
{
    return mFiltered;
}

inline void GWRRobust::setFiltered(bool value)
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

}

#endif // GWRROBUST_H