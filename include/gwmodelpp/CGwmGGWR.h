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

    GwmGGWRDiagnostic(const arma::vec &diag)
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

    GwmGLMDiagnostic(const arma::vec &diag)
    {
        AIC = diag(0);
        AICc = diag(1);
        NullDev = diag(2);
        Dev = diag(3);
        RSquare = diag(4);
    }
};

class CGwmGGWR : public CGwmGWRBase, public IGwmBandwidthSelectable, public IGwmOpenmpParallelizable
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

    static std::map<std::string, double> TolUnitDict;
    static void initTolUnitDict();

    typedef double (CGwmGGWR::*BandwidthSelectCriterionFunction)(CGwmBandwidthWeight *);
    typedef arma::mat (CGwmGGWR::*GGWRfitFunction)(const arma::mat& x, const arma::vec& y);
    typedef arma::vec (CGwmGGWR::*CalWtFunction)(const arma::mat &x, const arma::vec &y, arma::mat w);

    typedef std::tuple<std::string, arma::mat, NameFormat> CreateResultLayerData;

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
    int parallelAbility() const override;

    ParallelType parallelType() const override;
    void setParallelType(const ParallelType &type) override;

public: // IOpenmpParallelable interface
    void setOmpThreadNum(const int threadNum) override;

public:
    static arma::vec gwReg(const arma::mat &x, const arma::vec &y, const arma::vec &w);

    static arma::vec gwRegHatmatrix(const arma::mat &x, const arma::vec &y, const arma::vec &w, arma::uword focus, arma::mat &ci, arma::mat &s_ri);

    static arma::mat dpois(arma::mat y, arma::mat mu);
    static arma::mat dbinom(arma::mat y, arma::mat m, arma::mat mu);
    static arma::mat lchoose(arma::mat n, arma::mat k);
    static arma::mat lgammafn(arma::mat x);

    static arma::mat CiMat(const arma::mat &x, const arma::vec &w);

    arma::mat predict(const arma::mat& locations) override;
    arma::mat fit() override;
    arma::mat fit(const arma::mat &x, const arma::vec &y, arma::mat &betasSE, arma::vec &shat, arma::vec &qdiag, arma::mat &S);

protected:
    arma::mat fitPoissonSerial(const arma::mat& x, const arma::vec& y);
    arma::mat fitBinomialSerial(const arma::mat& x, const arma::vec& y);
#ifdef ENABLE_OPENMP
    arma::mat fitPoissonOmp(const arma::mat& x, const arma::vec& y);
    arma::mat fitBinomialOmp(const arma::mat& x, const arma::vec& y);
#endif
    arma::mat diag(arma::mat a);

    arma::vec PoissonWtSerial(const arma::mat &x, const arma::vec &y, arma::mat w);
    arma::vec BinomialWtSerial(const arma::mat &x, const arma::vec &y, arma::mat w);
#ifdef ENABLE_OPENMP
    arma::vec PoissonWtOmp(const arma::mat &x, const arma::vec &y, arma::mat w);
    arma::vec BinomialWtOmp(const arma::mat &x, const arma::vec &y, arma::mat w);
#endif
    void CalGLMModel(const arma::mat& x, const arma::vec& y);
    // todo: QStringLiteral 用法不确定
    void createResultLayer(std::initializer_list<CreateResultLayerData> items);

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
    size_t getMaxiter() const;

    arma::mat getWtMat1() const;
    arma::mat getWtMat2() const;

    GwmGGWRDiagnostic getDiagnostic() const;
    GwmGLMDiagnostic getGLMDiagnostic() const;

    bool setFamily(Family family);
    void setTol(double tol, std::string unit);
    void setMaxiter(std::size_t maxiter);

    void setBandwidthSelectionCriterionType(const BandwidthSelectionCriterionType &bandwidthSelectionCriterionType);
    BandwidthCriterionList bandwidthSelectorCriterions() const;
    BandwidthCriterionList mBandwidthSelectionCriterionList;
    BandwidthSelectionCriterionType bandwidthSelectionCriterionType() const;

    bool autoselectBandwidth() const;
    void setIsAutoselectBandwidth(bool value);

    arma::mat regressionData() const;
    void setRegressionData(const arma::mat &locations);

    bool hasHatMatrix() const;
    void setHasHatMatrix(bool value);

    bool hasRegressionData() const;
    void setHasRegressionData(bool value);

    //子节点命名记录标
    static int treeChildCount;

protected:
    Family mFamily;
    double mTol=1e-5;
    std::string mTolUnit;
    std::size_t mMaxiter=20;

    bool mHasHatMatrix = true;
    bool mHasRegressionData = false;

    arma::mat mBetasSE;

    arma::vec mShat;
    arma::mat mS;
    double mGwDev;

    arma::mat mRegressionData;

    arma::mat mWtMat1;
    arma::mat mWtMat2;

    GwmGGWRDiagnostic mDiagnostic;
    GwmGLMDiagnostic mGLMDiagnostic;
    CreateResultLayerData mResultList;

    arma::mat mWt2;
    arma::mat myAdj;

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

inline size_t CGwmGGWR::getMaxiter() const
{
    return mMaxiter;
}

inline arma::mat CGwmGGWR::getWtMat1() const
{
    return mWtMat1;
}

inline arma::mat CGwmGGWR::getWtMat2() const
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

inline void CGwmGGWR::setTol(double tol, std::string unit)
{
    mTolUnit = unit;
    mTol = double(tol) * TolUnitDict[unit];
}

inline void CGwmGGWR::setMaxiter(size_t maxiter)
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
inline arma::mat CGwmGGWR::regressionData() const
{
    return mRegressionData;
}

inline void CGwmGGWR::setRegressionData(const arma::mat &locations)
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

// inline arma::mat CGwmGGWR::fit(const arma::mat &x, const arma::vec &y, arma::mat &betasSE, arma::vec &shat, arma::vec &qdiag, arma::mat &S)
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