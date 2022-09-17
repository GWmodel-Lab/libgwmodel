#ifndef CGWMMGWR_H
#define CGWMMGWR_H

#include "CGwmSpatialAlgorithm.h"
#include "spatialweight/CGwmSpatialWeight.h"
#include "IGwmRegressionAnalysis.h"
#include "IGwmParallelizable.h"
#include "IGwmBandwidthSelectable.h"


class CGwmMGWR : public CGwmSpatialAlgorithm, public IGwmRegressionAnalysis, public IGwmBandwidthSelectable, public IGwmOpenmpParallelizable
{
public:
    enum BandwidthInitilizeType
    {
        Null,
        Initial,
        Specified
    };
    //static GwmEnumValueNameMapper<BandwidthInitilizeType> BandwidthInitilizeTypeNameMapper;
    static unordered_map<BandwidthInitilizeType, string> BandwidthInitilizeTypeNameMapper;

    enum BandwidthSelectionCriterionType
    {
        CV,
        AIC
    };
    //static GwmEnumValueNameMapper<BandwidthSelectionCriterionType> BandwidthSelectionCriterionTypeNameMapper;
    static unordered_map<BandwidthSelectionCriterionType, string> BandwidthSelectionCriterionTypeNameMapper;

    enum BackFittingCriterionType
    {
        CVR,
        dCVR
    };
    //static GwmEnumValueNameMapper<BackFittingCriterionType> BackFittingCriterionTypeNameMapper;
    static unordered_map<BackFittingCriterionType, string> BackFittingCriterionTypeNameMapper;

    //CGwmGWRBasic::OLSVar mOLSVar;

    //CGwmGWRBasic::OLSVar CalOLS(const mat &x, const vec &y);


    typedef double (CGwmMGWR::*BandwidthSizeCriterionFunction)(CGwmBandwidthWeight*);
    typedef mat (CGwmMGWR::*RegressionAllFunction)(const arma::mat&, const arma::vec&);
    typedef vec (CGwmMGWR::*RegressionVarFunction)(const arma::vec&, const arma::vec&, int, mat&);
    //typedef QPair<QString, const mat> CreateResultLayerDataItem;

private:
    static vec Fitted(const mat& x, const mat& betas)
    {
        return sum(betas % x, 1);
    }

    static double RSS(const mat& x, const mat& y, const mat& betas)
    {
        vec r = y - Fitted(x, betas);
        return sum(r % r);
    }

    static double AICc(const mat& x, const mat& y, const mat& betas, const vec& shat)
    {
        double ss = RSS(x, y, betas), n = x.n_rows;
        return n * log(ss / n) + n * log(2 * datum::pi) + n * ((n + shat(0)) / (n - 2 - shat(0)));
    }

    static GwmRegressionDiagnostic CalcDiagnostic(const mat& x, const vec& y, const mat& S0, double RSS0);
    

public:
    CGwmMGWR();

public:
    void run() override;

    bool mOLS = true;
    bool OLS() const;
    void setOLS(bool value);

    //CGwmGWRBasic::OLSVar getOLSVar() const;

    BandwidthInitilizeType bandwidthInitilize() const;
    void setBandwidthInitilize(const BandwidthInitilizeType &bandwidthInitilize);

    BandwidthSelectionCriterionType bandwidthSelectionApproach() const;
    void setBandwidthSelectionApproach(const BandwidthSelectionCriterionType &bandwidthSelectionApproach);

    bool preditorCentered() const;
    void setPreditorCentered(const bool &preditorCentered);

    double bandwidthSelectThreshold() const;
    void setBandwidthSelectThreshold(const double &bandwidthSelectThreshold);

    bool hasHatMatrix() const;
    void setHasHatMatrix(bool hasHatMatrix);

    int bandwidthSelectRetryTimes() const;
    void setBandwidthSelectRetryTimes(int bandwidthSelectRetryTimes);

    int maxIteration() const;
    void setMaxIteration(int maxIteration);

    BackFittingCriterionType criterionType() const;
    void setCriterionType(const BackFittingCriterionType &criterionType);

    double criterionThreshold() const;
    void setCriterionThreshold(double criterionThreshold);

    int adaptiveLower() const;
    void setAdaptiveLower(int adaptiveLower);

    mat betas() const;

    bool hasRegressionLayer()
    {
        return mRegressionLayer != nullptr;
    }


public:     // GwmTaskThread interface
    string name() const  { return tr("Multiscale GWR"); }


public:     // GwmSpatialAlgorithm interface
    bool isValid() override;


public:     // GwmSpatialMultiscaleAlgorithm interface
    virtual void setSpatialWeights(const CGwmSpatialWeight &spatialWeights);


public:     // IBandwidthSizeSelectable interface
    double criterion(CGwmBandwidthWeight *weight) //override
    {
        return (this->*mBandwidthSizeCriterion)(weight);
    }


public:     // IRegressionAnalysis interface
    GwmVariable dependentVariable() const override;

    void setDependentVariable(const GwmVariable &variable) override;

    GwmVariable independentVariables() const; //override;

    void setIndependentVariables(const GwmVariable &variables);// override;

    GwmRegressionDiagnostic diagnostic() const override;

    mat regression(const mat &x, const vec &y) override;


public:     // IParallelalbe interface
    int parallelAbility() const override;
    ParallelType parallelType() const override;
    void setParallelType(const ParallelType &type) override;


public:     // IOpenmpParallelable interface
    void setOmpThreadNum(const int threadNum) override;

    void setCanceled(bool canceled);

protected:
    void initPoints();
    void initXY(mat& x, mat& y, const GwmVariable& depVar, const GwmVariable& indepVars);

    CGwmBandwidthWeight* bandwidth(int i)
    {
        return static_cast<CGwmBandwidthWeight*>(mSpatialWeights[i].weight());
    }

    mat regressionAllSerial(const mat& x, const vec& y);
#ifdef ENABLE_OpenMP
    mat regressionAllOmp(const mat& x, const vec& y);
#endif
    vec regressionVarSerial(const vec& x, const vec& y, const int var, mat& S);
#ifdef ENABLE_OpenMP
    vec regressionVarOmp(const vec& x, const vec& y, const int var, mat& S);
#endif

    BandwidthSizeCriterionFunction bandwidthSizeCriterionAll(BandwidthSelectionCriterionType type);
    double mBandwidthSizeCriterionAllCVSerial(CGwmBandwidthWeight* bandwidthWeight);
#ifdef ENABLE_OpenMP
    double mBandwidthSizeCriterionAllCVOmp(GwmBandwidthWeight* bandwidthWeight);
#endif
    double mBandwidthSizeCriterionAllAICSerial(CGwmBandwidthWeight* bandwidthWeight);
#ifdef ENABLE_OpenMP
    double mBandwidthSizeCriterionAllAICOmp(GwmBandwidthWeight* bandwidthWeight);
#endif

    BandwidthSizeCriterionFunction bandwidthSizeCriterionVar(BandwidthSelectionCriterionType type);
    double mBandwidthSizeCriterionVarCVSerial(CGwmBandwidthWeight* bandwidthWeight);
#ifdef ENABLE_OpenMP
    double mBandwidthSizeCriterionVarCVOmp(GwmBandwidthWeight* bandwidthWeight);
#endif
    double mBandwidthSizeCriterionVarAICSerial(CGwmBandwidthWeight* bandwidthWeight);
#ifdef ENABLE_OpenMP
    double mBandwidthSizeCriterionVarAICOmp(GwmBandwidthWeight* bandwidthWeight);
#endif

    void createResultLayer(initializer_list<CreateResultLayerDataItem> data);

protected:
    CGwmBandwidthSelector selector;
private:
    QgsVectorLayer* mRegressionLayer = nullptr;
    mat mDataPoints;
    mat mRegressionPoints;

    RegressionAllFunction mRegressionAll = &CGwmMGWR::regressionAllSerial;
    RegressionVarFunction mRegressionVar = &CGwmMGWR::regressionVarSerial;

    GwmVariable mDepVar;
    GwmVariable mIndepVars;

    CGwmSpatialWeight mInitSpatialWeight;
    BandwidthSizeCriterionFunction mBandwidthSizeCriterion = &CGwmMGWR::mBandwidthSizeCriterionAllCVSerial;
    int mBandwidthSelectionCurrentIndex = 0;

    BandwidthInitilizeType mBandwidthInitilize;
    BandwidthSelectionCriterionType mBandwidthSelectionApproach;
    bool mPreditorCentered;
    double mBandwidthSelectThreshold;
    uword mBandwidthSelectRetryTimes = 5;
    int mMaxIteration = 500;
    BackFittingCriterionType mCriterionType = BackFittingCriterionType::CVR;
    double mCriterionThreshold = 1e-6;
    int mAdaptiveLower = 10;

    bool mHasHatMatrix = true;

    mat mX;
    vec mY;
    mat mBetas;
    mat mBetasSE;
    mat mBetasTV;

    mat mS0;
    cube mSArray;
    cube mC;
    mat mX0;
    vec mY0;
    vec mXi;
    vec mYi;

    double mRSS0;

    GwmRegressionDiagnostic mDiagnostic;

    ParallelType mParallelType = ParallelType::SerialOnly;
    int mOmpThreadNum = 8;

public:
    static int treeChildCount;
};



inline GwmVariable CGwmMGWR::dependentVariable() const
{
    return mDepVar;
}

inline void CGwmMGWR::setDependentVariable(const GwmVariable &variable)
{
    mDepVar = variable;
}

inline GwmVariable CGwmMGWR::independentVariables() const
{
    return mIndepVars;
}

inline void CGwmMGWR::setIndependentVariables(const GwmVariable &variables)
{
    mIndepVars = variables;
}

inline GwmRegressionDiagnostic CGwmMGWR::diagnostic() const
{
    return mDiagnostic;
}

inline int CGwmMGWR::adaptiveLower() const
{
    return mAdaptiveLower;
}

inline void CGwmMGWR::setAdaptiveLower(int adaptiveLower)
{
    mAdaptiveLower = adaptiveLower;
}

inline double CGwmMGWR::criterionThreshold() const
{
    return mCriterionThreshold;
}

inline void CGwmMGWR::setCriterionThreshold(double criterionThreshold)
{
    mCriterionThreshold = criterionThreshold;
}

inline CGwmMGWR::BackFittingCriterionType CGwmMGWR::criterionType() const
{
    return mCriterionType;
}

inline void CGwmMGWR::setCriterionType(const BackFittingCriterionType &criterionType)
{
    mCriterionType = criterionType;
}

inline int CGwmMGWR::maxIteration() const
{
    return mMaxIteration;
}

inline void CGwmMGWR::setMaxIteration(int maxIteration)
{
    mMaxIteration = maxIteration;
}

inline int CGwmMGWR::bandwidthSelectRetryTimes() const
{
    return mBandwidthSelectRetryTimes;
}

inline void CGwmMGWR::setBandwidthSelectRetryTimes(int bandwidthSelectRetryTimes)
{
    mBandwidthSelectRetryTimes = bandwidthSelectRetryTimes;
}

inline bool CGwmMGWR::preditorCentered() const
{
    return mPreditorCentered;
}

inline void CGwmMGWR::setPreditorCentered(const bool &preditorCentered)
{
    mPreditorCentered = preditorCentered;
}

inline CGwmMGWR::BandwidthSelectionCriterionType CGwmMGWR::bandwidthSelectionApproach() const
{
    return CGwmMGWR::mBandwidthSelectionApproach;
}

inline void CGwmMGWR::setBandwidthSelectionApproach(const BandwidthSelectionCriterionType &bandwidthSelectionApproach)
{
    mBandwidthSelectionApproach = bandwidthSelectionApproach;
}

inline CGwmMGWR::BandwidthInitilizeType CGwmMGWR::bandwidthInitilize() const
{
    return CGwmMGWR::mBandwidthInitilize;
}

inline void CGwmMGWR::setBandwidthInitilize(const BandwidthInitilizeType &bandwidthInitilize)
{
    mBandwidthInitilize = bandwidthInitilize;
}

inline double CGwmMGWR::bandwidthSelectThreshold() const
{
    return mBandwidthSelectThreshold;
}

inline void CGwmMGWR::setBandwidthSelectThreshold(const double &bandwidthSelectThreshold)
{
    mBandwidthSelectThreshold = bandwidthSelectThreshold;
}

inline bool CGwmMGWR::hasHatMatrix() const
{
    return mHasHatMatrix;
}

inline void CGwmMGWR::setHasHatMatrix(bool hasHatMatrix)
{
    mHasHatMatrix = hasHatMatrix;
}

inline mat CGwmMGWR::betas() const
{
    return mBetas;
}

inline int CGwmMGWR::parallelAbility() const
{
    return ParallelType::SerialOnly
        #ifdef ENABLE_OpenMP
            | IParallelalbe::OpenMP
        #endif
            ;
}

inline ParallelType CGwmMGWR::parallelType() const
{
    return mParallelType;
}

inline void CGwmMGWR::setOmpThreadNum(const int threadNum)
{
    mOmpThreadNum = threadNum;
}

inline bool CGwmMGWR::OLS() const
{
    return mOLS;
}

/*inline CGwmGWRBasic::OLSVar CGwmMGWR::getOLSVar() const
{
    return mOLSVar;
}*/

inline void CGwmMGWR::setOLS(bool value)
{
    mOLS = value;
}


#endif // GWMMULTISCALEGWRTASKTHREAD_H
