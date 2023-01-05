#ifndef CGWMLOCALCOLLINEARITYGWR_H
#define GWMLOCALCOLLINEARITYGWR_H

#include <utility>
#include <string>
#include <initializer_list>
#include "CGwmGWRBase.h"
#include "GwmRegressionDiagnostic.h"
#include "IGwmBandwidthSelectable.h"
#include "IGwmVarialbeSelectable.h"
#include "IGwmParallelizable.h"
#include <exception>

using namespace std;

class CGwmLocalCollinearityGWR : public CGwmGWRBase, public IGwmBandwidthSelectable, public IGwmOpenmpParallelizable
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
    
    typedef tuple<string, mat, NameFormat> ResultLayerDataItem;
    typedef double (CGwmLocalCollinearityGWR::*BandwidthSelectionCriterionCalculator)(CGwmBandwidthWeight*);
    typedef mat (CGwmLocalCollinearityGWR::*RegressionCalculator)(const mat&, const vec&);

    static GwmRegressionDiagnostic CalcDiagnostic(const mat& x, const vec& y, const mat& betas, const vec& shat);

public:
    CGwmLocalCollinearityGWR();
    ~CGwmLocalCollinearityGWR();

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
    mat predictData() const
    {
        return mPredictData;
    }
    void setPredictData(mat &value)
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

    double getCriterion(CGwmBandwidthWeight* weight)
    {
        return (this->*mBandwidthSelectionCriterionFunction)(weight);
    }

    double getCriterion(const vector<size_t>& variables)
    {
        throw std::runtime_error("not available"); 

    }


public:
    mat fit() override;

public:
    mat predict(const mat& x, const vec& y) 
    {
        return (this->*mPredictFunction)(x, y);
    }
    mat predict(const mat &locations) override
    {
        throw std::runtime_error("not available"); 
    }
    mat fit(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qdiag, mat& S) 
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
    BandwidthSelectionCriterionCalculator mBandwidthSelectionCriterionFunction = &CGwmLocalCollinearityGWR::bandwidthSizeCriterionCVSerial;


    //返回cv的函数
    double LcrCV(double bw,uword kernel, bool adaptive,double lambda,bool lambdaAdjust,double cnThresh);
    //ridge.lm函数
    vec ridgelm(const vec& w,double lambda);


private:
    double bandwidthSizeCriterionCVSerial(CGwmBandwidthWeight* bandwidthWeight);
    
    double mLambda=0;
    bool mLambdaAdjust=false;
    double mCnThresh=30;
    mat mPredictData;
    bool mHasHatMatrix = false;
    bool mHasPredict=false;
    bool mIsAutoselectBandwidth = false;
    double mTrS = 0;
    double mTrStS = 0;
    vec mSHat;

public:
    mat predictSerial(const mat& x, const vec& y);
#ifdef ENABLE_OPENMP
    mat predictOmp(const mat& x, const vec& y);
#endif
#ifdef ENABLE_OPENMP
    double bandwidthSizeCriterionCVOmp(CGwmBandwidthWeight* bandwidthWeight);
#endif
    RegressionCalculator mPredictFunction = &CGwmLocalCollinearityGWR::predictSerial;
    ParallelType mParallelType = ParallelType::SerialOnly;

    uword mOmpThreadNum = 8;
    uword mGpuId = 0;
    uword mGroupSize = 64;
};

inline int CGwmLocalCollinearityGWR::parallelAbility() const
{
    return ParallelType::SerialOnly | ParallelType::OpenMP;
}

inline ParallelType CGwmLocalCollinearityGWR::parallelType() const
{
    return mParallelType;
}

inline void CGwmLocalCollinearityGWR::setOmpThreadNum(const int threadNum)
{
    mOmpThreadNum = threadNum;
}
#endif  //CGWMLOCALCOLLINEARITYGWR_H