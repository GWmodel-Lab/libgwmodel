#ifndef CGWMGGWR_H
#define CGWMGGWR_H

#include <utility>
#include <string>
#include <map>
#include <tuple>
#include <vector>
#include <initializer_list>
#include "CGwmGWRBase.h"
#include "GwmRegressionDiagnostic.h"
#include "IGwmBandwidthSelectable.h"
#include "IGwmVarialbeSelectable.h"
#include "IGwmParallelizable.h"
#include "CGwmGWRBasic.h"
#include "CGwmBandwidthSelector.h"
#include "CGwmBandwidthSelector.h"

struct GwmGGWRDiagnostic
{
    double RSS;
    double AIC;
    double AICc;
    double RSquare;

    GwmGGWRDiagnostic()
    {
        AIC = 0.0;
        AICc = 0.0;
        RSS = 0.0;
        RSquare = 0.0;
    }

    GwmGGWRDiagnostic(const vec &diag)
    {
        AIC = diag(0);
        AICc = diag(1);
        RSS = diag(2);
        RSquare = diag(3);
    }
};

struct GwmGLMDiagnostic
{
    double NullDev;
    double Dev;
    double AIC;
    double AICc;
    double RSquare;

    GwmGLMDiagnostic()
    {
        AIC = 0.0;
        AICc = 0.0;
        Dev = 0.0;
        NullDev = 0.0;
        RSquare = 0.0;
    }

    GwmGLMDiagnostic(const vec &diag)
    {
        AIC = diag(0);
        AICc = diag(1);
        NullDev = diag(2);
        Dev = diag(3);
        RSquare = diag(4);
    }
};

class CGwmGGWR : public CGwmGWRBase, public IGwmBandwidthSelectable, public IGwmParallelizable
{
public:
    enum Family
    {
        Poisson,
        Binomial
    };

    enum BandwidthSelectionCriterionType
    {
        AIC,
        CV
    };

    enum NameFormat
    {
        Fixed,
        VarName,
        PrefixVarName,
        SuffixVariable
    };

    static map<string, double> TolUnitDict;
    static void initTolUnitDict();

    typedef double (CGwmGGWR::*BandwidthSelectCriterionFunction)(CGwmBandwidthWeight *);
    typedef mat (CGwmGGWR::*GGWRfitFunction)(const mat& x, const vec& y);
    typedef mat (CGwmGGWR::*CalWtFunction)(const mat &x, const vec &y, mat w);

    typedef tuple<string, mat, NameFormat> CreateResultLayerData;

public:
    CGwmGGWR(){};
    ~CGwmGGWR(){};

public: // IBandwidthSizeSelectable interface
    double getCriterion(CGwmBandwidthWeight *bandwidthWeight) override
    {
        return (this->*mBandwidthSelectCriterionFunction)(bandwidthWeight);
    }

public: // IRegressionAnalysis interface
   /*  arma::mat predict(const arma::mat &x, const arma::vec &y)
    {
        return (this->*mGGWRfitFunction)(x, y);
    } */

public: // IParallelalbe interface
    int parallelAbility() const;

    ParallelType parallelType() const;
    void setParallelType(const ParallelType &type) override;

public: // IOpenmpParallelable interface
    void setOmpThreadNum(const int threadNum);

public:
    static vec gwReg(const mat &x, const vec &y, const vec &w, int focus);

    static vec gwRegHatmatrix(const mat &x, const vec &y, const vec &w, int focus, mat &ci, mat &s_ri);

    static mat dpois(mat y, mat mu);
    static mat dbinom(mat y, mat m, mat mu);
    static mat lchoose(mat n, mat k);
    static mat lgammafn(mat x);

    static mat CiMat(const mat &x, const vec &w);

    mat predict(const mat& locations) override;
    mat fit() override;
    mat fit(const mat &x, const vec &y, mat &betasSE, vec &shat, vec &qdiag, mat &S);

protected:
    mat fitPoissonSerial(const mat& x, const vec& y);
    mat fitBinomialSerial(const mat& x, const vec& y);
#ifdef ENABLE_OPENMP
    mat fitPoissonOmp(const mat& x, const vec& y);
    mat fitBinomialOmp(const mat& x, const vec& y);
#endif
    mat diag(mat a);

    mat PoissonWtSerial(const mat &x, const vec &y, mat w);
    mat BinomialWtSerial(const mat &x, const vec &y, mat w);
#ifdef ENABLE_OPENMP
    mat PoissonWtOmp(const mat &x, const vec &y, mat w);
    mat BinomialWtOmp(const mat &x, const vec &y, mat w);
#endif
    void CalGLMModel(const mat& x, const vec& y);
    // todo: QStringLiteral 用法不确定
    void createResultLayer(initializer_list<CreateResultLayerData> items);

private:
    double bandwidthSizeGGWRCriterionCVSerial(CGwmBandwidthWeight *bandwidthWeight);
    double bandwidthSizeGGWRCriterionAICSerial(CGwmBandwidthWeight *bandwidthWeight);
#ifdef ENABLE_OPENMP
    double bandwidthSizeGGWRCriterionCVOmp(CGwmBandwidthWeight *bandwidthWeight);
    double bandwidthSizeGGWRCriterionAICOmp(CGwmBandwidthWeight *bandwidthWeight);
#endif
public:
    Family getFamily() const;
    double getTol() const;
    int getMaxiter() const;

    mat getWtMat1() const;
    mat getWtMat2() const;

    GwmGGWRDiagnostic getDiagnostic() const;
    GwmGLMDiagnostic getGLMDiagnostic() const;

    bool setFamily(Family family);
    void setTol(double tol, string unit);
    void setMaxiter(int maxiter);

    void setBandwidthSelectionCriterionType(const BandwidthSelectionCriterionType &bandwidthSelectionCriterionType);
    BandwidthCriterionList bandwidthSelectorCriterions() const;
    BandwidthCriterionList mBandwidthSelectionCriterionList;
    BandwidthSelectionCriterionType bandwidthSelectionCriterionType() const;

    bool autoselectBandwidth() const;
    void setIsAutoselectBandwidth(bool value);

    mat regressionData() const;
    void setRegressionData(const mat &locations);

    bool hasHatMatrix() const;
    void setHasHatMatrix(bool value);

    bool hasRegressionData() const;
    void setHasRegressionData(bool value);

    //子节点命名记录标
    static int treeChildCount;

protected:
    Family mFamily;
    double mTol=1e-5;
    string mTolUnit;
    int mMaxiter=20;

    bool mHasHatMatrix = true;
    bool mHasRegressionData = false;

    mat mBetasSE;

    vec mShat;
    mat mS;
    double mGwDev;

    mat mRegressionData;

    mat mWtMat1;
    mat mWtMat2;

    GwmGGWRDiagnostic mDiagnostic;
    GwmGLMDiagnostic mGLMDiagnostic;
    CreateResultLayerData mResultList;

    mat mWt2;
    mat myAdj;

    double mLLik = 0;

    GGWRfitFunction mGGWRfitFunction = &CGwmGGWR::fitPoissonSerial;
    CalWtFunction mCalWtFunction = &CGwmGGWR::PoissonWtSerial;

    bool mIsAutoselectBandwidth = false;
    BandwidthSelectionCriterionType mBandwidthSelectionCriterionType = BandwidthSelectionCriterionType::AIC;
    BandwidthSelectCriterionFunction mBandwidthSelectCriterionFunction = &CGwmGGWR::bandwidthSizeGGWRCriterionCVSerial;
    CGwmBandwidthSelector mBandwidthSizeSelector;

    ParallelType mParallelType = ParallelType::SerialOnly;
    int mOmpThreadNum = 8;

    //void createDistanceParameter();
    void createPredictionDistanceParameter(const arma::mat& locations);

    //DistanceParameter *mRegressionDistanceParameter = nullptr;
    //DistanceParameter *mPredictionDistanceParameter = nullptr;
};

inline CGwmGGWR::Family CGwmGGWR::getFamily() const
{
    return mFamily;
}

inline double CGwmGGWR::getTol() const
{
    return mTol;
}

inline int CGwmGGWR::getMaxiter() const
{
    return mMaxiter;
}

inline mat CGwmGGWR::getWtMat1() const
{
    return mWtMat1;
}

inline mat CGwmGGWR::getWtMat2() const
{
    return mWtMat2;
}

inline GwmGGWRDiagnostic CGwmGGWR::getDiagnostic() const
{
    return mDiagnostic;
}

inline GwmGLMDiagnostic CGwmGGWR::getGLMDiagnostic() const
{
    return mGLMDiagnostic;
}

inline void CGwmGGWR::setTol(double tol, string unit)
{
    mTolUnit = unit;
    mTol = double(tol) * TolUnitDict[unit];
}

inline void CGwmGGWR::setMaxiter(int maxiter)
{
    mMaxiter = maxiter;
}

inline BandwidthCriterionList CGwmGGWR::bandwidthSelectorCriterions() const
{
    return mBandwidthSizeSelector.bandwidthCriterion();
}

inline bool CGwmGGWR::hasHatMatrix() const
{
    return mHasHatMatrix;
}

inline void CGwmGGWR::setHasHatMatrix(bool value)
{
    mHasHatMatrix = value;
}

inline bool CGwmGGWR::hasRegressionData() const
{
    return mHasRegressionData;
}

inline void CGwmGGWR::setHasRegressionData(bool value)
{
    mRegressionData = value;
}
inline mat CGwmGGWR::regressionData() const
{
    return mRegressionData;
}

inline void CGwmGGWR::setRegressionData(const mat &locations)
{
    mRegressionData = locations;
}

inline CGwmGGWR::BandwidthSelectionCriterionType CGwmGGWR::bandwidthSelectionCriterionType() const
{
    return mBandwidthSelectionCriterionType;
}

inline bool CGwmGGWR::autoselectBandwidth() const
{
    return mIsAutoselectBandwidth;
}

inline void CGwmGGWR::setIsAutoselectBandwidth(bool value)
{
    mIsAutoselectBandwidth = value;
}

inline int CGwmGGWR::parallelAbility() const
    {
        return ParallelType::SerialOnly
#ifdef ENABLE_OPENMP
            | ParallelType::OpenMP
#endif        
        ;
    }


inline ParallelType CGwmGGWR::parallelType() const
{
    return mParallelType;
}

inline void CGwmGGWR::setOmpThreadNum(const int threadNum)
{
    mOmpThreadNum = threadNum;
}

// inline mat CGwmGGWR::fit(const mat &x, const vec &y, mat &betasSE, vec &shat, vec &qdiag, mat &S)
// {
//     switch (type)
//     {
//     case CGwmGGWR::Family::Poisson:
//         return regressionPoissonSerial(x, y);
//         break;
//     case CGwmGGWR::Family::Binomial:
//         return regressionBinomialSerial(x, y);
//         break;
//     default:
//         return regressionPoissonSerial(x, y);
//         break;
//     }
// }

#endif // CGWMGGWR_H