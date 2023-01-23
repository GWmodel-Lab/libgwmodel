#ifndef GWRLOCALCOLLINEARITY_H
#define GWRLOCALCOLLINEARITY_H

#include <utility>
#include <string>
#include <initializer_list>
#include "GWRBase.h"
#include "RegressionDiagnostic.h"
#include "IBandwidthSelectable.h"
#include "IVarialbeSelectable.h"
#include "IParallelizable.h"
#include <exception>


namespace gwm
{

class GWRLocalCollinearity : public GWRBase, public IBandwidthSelectable, public IParallelizable, public IParallelOpenmpEnabled
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
    typedef double (GWRLocalCollinearity::*BandwidthSelectionCriterionCalculator)(BandwidthWeight*);
    typedef arma::mat (GWRLocalCollinearity::*PredictCalculator)(const arma::mat&, const arma::vec&);

    static RegressionDiagnostic CalcDiagnostic(const arma::mat& x, const arma::vec& y, const arma::mat& betas, const arma::vec& shat);

public:
    GWRLocalCollinearity();
    ~GWRLocalCollinearity();

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

    RegressionDiagnostic dialnostic() const
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

    double getCriterion(BandwidthWeight* weight) override
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
    BandwidthSelectionCriterionCalculator mBandwidthSelectionCriterionFunction = &GWRLocalCollinearity::bandwidthSizeCriterionCVSerial;


    //返回cv的函数
    double LcrCV(double bw,arma::uword kernel, bool adaptive,double lambda,bool lambdaAdjust,double cnThresh);
    //ridge.lm函数
    arma::vec ridgelm(const arma::vec& w,double lambda);


private:
    double bandwidthSizeCriterionCVSerial(BandwidthWeight* bandwidthWeight);
#ifdef ENABLE_OPENMP
    double bandwidthSizeCriterionCVOmp(BandwidthWeight* bandwidthWeight);
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

    PredictCalculator mPredictFunction = &GWRLocalCollinearity::predictSerial;
    ParallelType mParallelType = ParallelType::SerialOnly;

    int mOmpThreadNum = 8;
    arma::uword mGpuId = 0;
    arma::uword mGroupSize = 64;
};

inline int GWRLocalCollinearity::parallelAbility() const
{
    return ParallelType::SerialOnly | ParallelType::OpenMP;
}

inline ParallelType GWRLocalCollinearity::parallelType() const
{
    return mParallelType;
}

inline void GWRLocalCollinearity::setOmpThreadNum(const int threadNum)
{
    mOmpThreadNum = threadNum;
}

}

#endif  //GWRLOCALCOLLINEARITY_H