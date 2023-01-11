#ifndef CGWMLOCALCOLLINEARITYGWR_H
#define CGWMLOCALCOLLINEARITYGWR_H

#include <utility>
#include <string>
#include <initializer_list>
#include "CGwmGWRBase.h"
#include "GwmRegressionDiagnostic.h"
#include "IGwmBandwidthSelectable.h"
#include "IGwmVarialbeSelectable.h"
#include "IGwmParallelizable.h"
#include <exception>


class CGwmGWRLocalCollinearity : public CGwmGWRBase, public IGwmBandwidthSelectable, public IGwmOpenmpParallelizable
{
public:
    enum BandwidthSelectionCriterionType
    {
        CV
    };

    enum NameFormat
    {
        Fixed,
        VarName,
        PrefixVarName,
        SuffixVariable
    };
    
    typedef std::tuple<std::string, arma::mat, NameFormat> ResultLayerDataItem;
    typedef double (CGwmGWRLocalCollinearity::*BandwidthSelectionCriterionCalculator)(CGwmBandwidthWeight*);
    typedef arma::mat (CGwmGWRLocalCollinearity::*PredictCalculator)(const arma::mat&, const arma::vec&);

    static GwmRegressionDiagnostic CalcDiagnostic(const arma::mat& x, const arma::vec& y, const arma::mat& betas, const arma::vec& shat);

public:
    CGwmGWRLocalCollinearity();
    ~CGwmGWRLocalCollinearity();

public:
    double cnThresh() const
    {
        return mCnThresh;
    }
    void setCnThresh(double cnThresh)
    {
        mCnThresh = cnThresh;
    }

    double lambda() const
    {
        return mLambda;
    }
    void setLambda(double lambda)
    {
        mLambda = lambda;
    }

    bool hasHatMatrix() const
    {
        return mHasHatMatrix;
    }
    void setHasHatMatrix(bool value)
    {
        mHasHatMatrix = value;
    }
    bool hasPredict() const
    {
        return mHasPredict;
    }
    void setHasPredict(bool value)
    {
        mHasPredict = value;
    }
    arma::mat predictData() const
    {
        return mPredictData;
    }
    void setPredictData(arma::mat &value)
    {
        mPredictData = value;
    }

    bool lambdaAdjust() const
    {
        return mLambdaAdjust;
    }
    void setLambdaAdjust(bool lambdaAdjust)
    {
        mLambdaAdjust = lambdaAdjust;
    }

    GwmRegressionDiagnostic dialnostic() const
    {
        return mDiagnostic;
    }

    bool isAutoselectBandwidth() const
    {
        return mIsAutoselectBandwidth;
    }

    void setIsAutoselectBandwidth(bool isAutoSelect)
    {
        mIsAutoselectBandwidth = isAutoSelect;
    }

    BandwidthCriterionList bandwidthSelectionCriterionList() const
    {
        return mBandwidthSelectionCriterionList;
    }

    BandwidthSelectionCriterionType bandwidthSelectionCriterion() const
    {
        return mBandwidthSelectionCriterion;
    }
    

    void setBandwidthSelectionCriterion(const BandwidthSelectionCriterionType& criterion);

    double getCriterion(CGwmBandwidthWeight* weight) override
    {
        return (this->*mBandwidthSelectionCriterionFunction)(weight);
    }

    /*double getCriterion(const vector<size_t>& variables) override
    {
        throw std::runtime_error("not available"); 

    }*/


public:
    arma::mat fit() override;

public:
    arma::mat predict(const arma::mat& x, const arma::vec& y) 
    {
        return (this->*mPredictFunction)(x, y);
    }
    arma::mat predict(const arma::mat &locations) override
    {
        throw std::runtime_error("not available"); 
    }
    arma::mat fit(const arma::mat& x, const arma::vec& y, arma::mat& betasSE, arma::vec& shat, arma::vec& qdiag, arma::mat& S) 
    {
        throw std::runtime_error("not available"); 
    }
    
    int parallelAbility() const override;
    ParallelType parallelType() const override;
    void setParallelType(const ParallelType& type) override;
    void setOmpThreadNum(const int threadNum) override;

protected:
    BandwidthCriterionList mBandwidthSelectionCriterionList;
    BandwidthSelectionCriterionType mBandwidthSelectionCriterion = BandwidthSelectionCriterionType::CV;
    BandwidthSelectionCriterionCalculator mBandwidthSelectionCriterionFunction = &CGwmGWRLocalCollinearity::bandwidthSizeCriterionCVSerial;


    //返回cv的函数
    double LcrCV(double bw,arma::uword kernel, bool adaptive,double lambda,bool lambdaAdjust,double cnThresh);
    //ridge.lm函数
    arma::vec ridgelm(const arma::vec& w,double lambda);


private:
    double bandwidthSizeCriterionCVSerial(CGwmBandwidthWeight* bandwidthWeight);
#ifdef ENABLE_OPENMP
    double bandwidthSizeCriterionCVOmp(CGwmBandwidthWeight* bandwidthWeight);
#endif

    double mLambda=0;
    bool mLambdaAdjust=false;
    double mCnThresh=30;
    arma::mat mPredictData;
    bool mHasHatMatrix = false;
    bool mHasPredict=false;
    bool mIsAutoselectBandwidth = false;
    double mTrS = 0;
    double mTrStS = 0;
    arma::vec mSHat;

public:
    arma::mat predictSerial(const arma::mat& x, const arma::vec& y);
#ifdef ENABLE_OPENMP
    arma::mat predictOmp(const arma::mat& x, const arma::vec& y);
#endif

    PredictCalculator mPredictFunction = &CGwmGWRLocalCollinearity::predictSerial;
    ParallelType mParallelType = ParallelType::SerialOnly;

    int mOmpThreadNum = 8;
    arma::uword mGpuId = 0;
    arma::uword mGroupSize = 64;
};

inline int CGwmGWRLocalCollinearity::parallelAbility() const
{
    return ParallelType::SerialOnly | ParallelType::OpenMP;
}

inline ParallelType CGwmGWRLocalCollinearity::parallelType() const
{
    return mParallelType;
}

inline void CGwmGWRLocalCollinearity::setOmpThreadNum(const int threadNum)
{
    mOmpThreadNum = threadNum;
}
#endif  //CGWMLOCALCOLLINEARITYGWR_H