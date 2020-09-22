#ifndef CGWMGWSS_H
#define CGWMGWSS_H

#include "CGwmSpatialMonoscaleAlgorithm.h"
#include "IGwmMultivariableAnalysis.h"
#include "IGwmParallelizable.h"

class CGwmGWSS : public CGwmSpatialMonoscaleAlgorithm, public IGwmMultivariableAnalysis, public IGwmOpenmpParallelizable
{
public:
    static double covwt(const mat &x1, const mat &x2, const vec &w)
    {
        return sum((sqrt(w) % (x1 - sum(x1 % w))) % (sqrt(w) % (x2 - sum(x2 % w)))) / (1 - sum(w % w));
    }

    static double corwt(const mat &x1, const mat &x2, const vec &w)
    {
        return covwt(x1,x2,w)/sqrt(covwt(x1,x1,w)*covwt(x2,x2,w));
    }

    static vec del(vec x,int rowcount);

    static vec rank(vec x)
    {
        vec n = linspace(0.0, (double)x.n_rows - 1, x.n_rows);
        vec res = n(sort_index(x));
        return n(sort_index(res)) + 1.0;
    }

    enum NameFormat
    {
        PrefixVarName,
        SuffixVarName,
        PrefixVarNamePair,
        SuffixVarNamePair
    };

    typedef tuple<string, mat, NameFormat> ResultLayerDataItem;

    typedef void (CGwmGWSS::*SummaryCalculator)();

protected:
    static vec findq(const mat& x, const vec& w);

public:
    CGwmGWSS();
    ~CGwmGWSS();

    bool quantile() const;
    void setQuantile(bool quantile);

    bool isCorrWithFirstOnly() const;
    void setIsCorrWithFirstOnly(bool corrWithFirstOnly);

    mat localMean() const { return mLocalMean; }
    mat localSDev() const { return mStandardDev; }
    mat localSkewness() const { return mLocalSkewness; }
    mat localCV() const { return mLCV; }
    mat localVar() const { return mLVar; }

    mat localMedian() const { return mLocalMedian; }
    mat iqr() const { return mIQR; }
    mat qi() const { return mQI; }

    mat localCov() const { return mCovmat; }
    mat localCorr() const { return mCorrmat; }
    mat localSCorr() const { return mSCorrmat; }

public:     // GwmAlgorithm interface;
    void run() override;

public:     // GwmSpatialAlgorithm interface
    bool isValid() override;

public:     // IGwmMultivariableAnalysis
    vector<GwmVariable> variables() const override;
    void setVariables(const vector<GwmVariable>& variables) override;

public:     // IGwmParallelizable
    int parallelAbility() const override;
    ParallelType parallelType() const override;
    void setParallelType(const ParallelType& type) override;

public:     // IGwmOpenmpParallelizable
    void setOmpThreadNum(const int threadNum) override;

private:
    void setXY(mat& x, const CGwmSimpleLayer* layer, const vector<GwmVariable>& variables);
    void createDistanceParameter();

    void summarySerial();
    void summaryOmp();

    void createResultLayer(vector<ResultLayerDataItem> items);

private:
    vector<GwmVariable> mVariables;

    bool mQuantile = false;
    bool mIsCorrWithFirstOnly = false;

    mat mX;
    mat mLocalMean;
    mat mStandardDev;
    mat mLocalSkewness;
    mat mLCV;
    mat mLVar;
    mat mLocalMedian;
    mat mIQR;
    mat mQI;
    mat mCovmat;
    mat mCorrmat;
    mat mSCorrmat;
    
    DistanceParameter* mDistanceParameter = nullptr;

    SummaryCalculator mSummaryFunction = &CGwmGWSS::summarySerial;
    
    ParallelType mParallelType = ParallelType::SerialOnly;
    int mOmpThreadNum = 8;
};

inline bool CGwmGWSS::quantile() const
{
    return mQuantile;
}

inline void CGwmGWSS::setQuantile(bool quantile)
{
    mQuantile = quantile;
}

inline bool CGwmGWSS::isCorrWithFirstOnly() const
{
    return mIsCorrWithFirstOnly;
}

inline void CGwmGWSS::setIsCorrWithFirstOnly(bool corrWithFirstOnly)
{
    mIsCorrWithFirstOnly = corrWithFirstOnly;
}

inline vector<GwmVariable> CGwmGWSS::variables() const
{
    return mVariables;
}

inline void CGwmGWSS::setVariables(const vector<GwmVariable>& variables)
{
    mVariables = variables;
}

inline int CGwmGWSS::parallelAbility() const
{
    return ParallelType::SerialOnly
#ifdef ENABLE_OPENMP
        | ParallelType::OpenMP
#endif        
        ;
}

inline ParallelType CGwmGWSS::parallelType() const
{
    return mParallelType;
}

inline void CGwmGWSS::setOmpThreadNum(const int threadNum)
{
    mOmpThreadNum = threadNum;
}

#endif  // CGWMGWSS_H