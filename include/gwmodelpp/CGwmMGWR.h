#ifndef CGWMMGWR_H
#define CGWMMGWR_H

#include <utility>
#include <string>
#include <initializer_list>
#include "CGwmSpatialMultiscaleAlgorithm.h"
#include "spatialweight/CGwmSpatialWeight.h"
#include "IGwmRegressionAnalysis.h"
#include "IGwmBandwidthSelectable.h"
#include "CGwmBandwidthSelector.h"
#include "IGwmParallelizable.h"


class CGwmMGWR : public CGwmSpatialMultiscaleAlgorithm, public IGwmBandwidthSelectable,public IGwmOpenmpParallelizable,public IGwmRegressionAnalysis 
{
public:
    enum BandwidthInitilizeType
    {
        Null,
        Initial,
        Specified
    };
    static unordered_map<BandwidthInitilizeType,string> BandwidthInitilizeTypeNameMapper;

    enum BandwidthSelectionCriterionType
    {
        CV,
        AIC
    };
    static unordered_map<BandwidthSelectionCriterionType,string> BandwidthSelectionCriterionTypeNameMapper;

    enum BackFittingCriterionType
    {
        CVR,
        dCVR
    };
    static unordered_map<BackFittingCriterionType,string> BackFittingCriterionTypeNameMapper;

    enum NameFormat
    {
        Fixed,
        VarName,
        PrefixVarName,
        SuffixVariable
    };
    //GwmBasicGWRAlgorithm::OLSVar mOLSVar;

    //GwmBasicGWRAlgorithm::OLSVar CalOLS(const mat &x, const vec &y);


    typedef double (CGwmMGWR::*BandwidthSizeCriterionFunction)(CGwmBandwidthWeight*);
    typedef mat (CGwmMGWR::*RegressionAllFunction)(const arma::mat&, const arma::vec&);
    typedef vec (CGwmMGWR::*RegressionVarFunction)(const arma::vec&, const arma::vec&, int, mat&);
    typedef mat (CGwmMGWR::*RegressionHatmatrixCalculator)(const mat&, const vec&, mat&, vec&, vec&, mat&);
    //typedef pair<string, const mat> CreateResultLayerDataItem;
    typedef tuple<string, mat, NameFormat> CreateResultLayerDataItem;
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
    bool isStoreS()
    {
        return mHasHatMatrix && (mSourceLayer->featureCount() < 8192);
    }

    static GwmRegressionDiagnostic CalcDiagnostic(const mat &x, const vec &y, const mat &S0, double RSS);

public:
    CGwmMGWR();

public:
    void run() override;

    bool mOLS = true;
    bool OLS() const;
    void setOLS(bool value);

    //GwmBasicGWRAlgorithm::OLSVar getOLSVar() const;

    vector<BandwidthInitilizeType> bandwidthInitilize() const;
    void setBandwidthInitilize(const vector<BandwidthInitilizeType> &bandwidthInitilize);

    vector<BandwidthSelectionCriterionType> bandwidthSelectionApproach() const;
    void setBandwidthSelectionApproach(const vector<BandwidthSelectionCriterionType> &bandwidthSelectionApproach);

    vector<bool> preditorCentered() const;
    void setPreditorCentered(const vector<bool> &preditorCentered);

    vector<double> bandwidthSelectThreshold() const;
    void setBandwidthSelectThreshold(const vector<double> &bandwidthSelectThreshold);

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
        return hasRegressionLayer()!= false;
    }

    BandwidthSizeCriterionFunction bandwidthSizeCriterionAll(BandwidthSelectionCriterionType type);
    BandwidthSizeCriterionFunction bandwidthSizeCriterionVar(BandwidthSelectionCriterionType type);


public:     // GwmTaskThread interface
    string name() const { return "Multiscale GWR"; }//override 


public:     // GwmSpatialAlgorithm interface
    bool isValid() override;


public:     // GwmSpatialMultiscaleAlgorithm interface
    virtual void setSpatialWeights(const vector<CGwmSpatialWeight> &spatialWeights);


public:     // IBandwidthSizeSelectable interface
    double getCriterion(CGwmBandwidthWeight* weight);


public:     // IRegressionAnalysis interface
    GwmVariable dependentVariable() const override;

    void setDependentVariable(const GwmVariable &variable) override;

    vector<GwmVariable> independentVariables() const override;

    void setIndependentVariables(const vector<GwmVariable> &variables) override;

    GwmRegressionDiagnostic diagnostic() const override;

    mat regression(const mat &x, const vec &y) override;

    mat regressionHatmatrixSerial(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qdiag, mat& S); 


public:     // IParallelalbe interface
    int parallelAbility() const override;
    ParallelType parallelType() const override;
    void setParallelType(const ParallelType &type) override;


public:     // IOpenmpParallelable interface
    void setOmpThreadNum(const int threadNum) override;

    //void setCanceled(bool canceled);



protected:
    //void initPoints();
    //void initXY(mat& x, mat& y, const GwmVariable& depVar, const vector<GwmVariable>& indepVars);
    virtual void setXY(mat& x, mat& y, const CGwmSimpleLayer* layer, const GwmVariable& depVar, const vector<GwmVariable>& indepVars);

    CGwmBandwidthWeight* bandwidth(int i)
    {
        return static_cast<CGwmBandwidthWeight*>(mSpatialWeights[i].weight());
    }
    mat regressionHatmatrix(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qdiag, mat& S)
    {
        return (this->*mRegressionHatmatrixFunction)(x, y, betasSE, shat, qdiag,S);
    }
    mat regressionAllSerial(const mat& x, const vec& y);

    mat regressionAllOmp(const mat& x, const vec& y);

    vec regressionVarSerial(const vec& x, const vec& y, const int var, mat& S);

    vec regressionVarOmp(const vec& x, const vec& y, const int var, mat& S);



    double mBandwidthSizeCriterionAllCVSerial(CGwmBandwidthWeight* bandwidthWeight);

    double mBandwidthSizeCriterionAllCVOmp(CGwmBandwidthWeight* bandwidthWeight);

    double mBandwidthSizeCriterionAllAICSerial(CGwmBandwidthWeight* bandwidthWeight);

    double mBandwidthSizeCriterionAllAICOmp(CGwmBandwidthWeight* bandwidthWeight);



    double mBandwidthSizeCriterionVarCVSerial(CGwmBandwidthWeight* bandwidthWeight);

    double mBandwidthSizeCriterionVarCVOmp(CGwmBandwidthWeight* bandwidthWeight);

    double mBandwidthSizeCriterionVarAICSerial(CGwmBandwidthWeight* bandwidthWeight);

    double mBandwidthSizeCriterionVarAICOmp(CGwmBandwidthWeight* bandwidthWeight);

    void createInitialDistanceParameter();

    void createResultLayer(initializer_list<CreateResultLayerDataItem> items);

protected:
    CGwmBandwidthSelector mselector;
private:
    //QgsVectorLayer* mRegressionLayer = nullptr;
    mat mDataPoints;
    mat mRegressionPoints;

    RegressionAllFunction mRegressionAll = &CGwmMGWR::regressionAllSerial;
    RegressionVarFunction mRegressionVar = &CGwmMGWR::regressionVarSerial;
    RegressionHatmatrixCalculator mRegressionHatmatrixFunction = &CGwmMGWR::regressionHatmatrixSerial;

    GwmVariable mDepVar;
    vector<GwmVariable> mIndepVars;

    CGwmSpatialWeight mInitSpatialWeight;
    BandwidthSizeCriterionFunction mBandwidthSizeCriterion = &CGwmMGWR::mBandwidthSizeCriterionAllCVSerial;
    int mBandwidthSelectionCurrentIndex = 0;


    vector<BandwidthInitilizeType> mBandwidthInitilize;
    vector<BandwidthSelectionCriterionType> mBandwidthSelectionApproach;
    vector<bool> mPreditorCentered;
    vector<double> mBandwidthSelectThreshold;
    uword mBandwidthSelectRetryTimes = 5;
    int mMaxIteration = 500;
    BackFittingCriterionType mCriterionType = BackFittingCriterionType::dCVR;
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

inline double CGwmMGWR::getCriterion(CGwmBandwidthWeight* weight)
{
    return (this->*mBandwidthSizeCriterion)(weight);
}

inline GwmVariable CGwmMGWR::dependentVariable() const
{
    return mDepVar;
}

inline void CGwmMGWR::setDependentVariable(const GwmVariable &variable)
{
    mDepVar = variable;
}

inline vector<GwmVariable> CGwmMGWR::independentVariables() const
{
    return mIndepVars;
}

inline void CGwmMGWR::setIndependentVariables(const vector<GwmVariable> &variables)
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

inline vector<bool> CGwmMGWR::preditorCentered() const
{
    return mPreditorCentered;
}

inline void CGwmMGWR::setPreditorCentered(const vector<bool> &preditorCentered)
{
    mPreditorCentered = preditorCentered;
}

inline vector<CGwmMGWR::BandwidthSelectionCriterionType> CGwmMGWR::bandwidthSelectionApproach() const
{
    return CGwmMGWR::mBandwidthSelectionApproach;
}

inline void CGwmMGWR::setBandwidthSelectionApproach(const vector<BandwidthSelectionCriterionType> &bandwidthSelectionApproach)
{
    if(bandwidthSelectionApproach.size()==mIndepVars.size()+1){
        mBandwidthSelectionApproach = bandwidthSelectionApproach;
    }
    else{

    }
    
}

inline vector<CGwmMGWR::BandwidthInitilizeType> CGwmMGWR::bandwidthInitilize() const
{
    return CGwmMGWR::mBandwidthInitilize;
}

inline void CGwmMGWR::setBandwidthInitilize(const vector<BandwidthInitilizeType> &bandwidthInitilize)
{
    if(bandwidthInitilize.size()==mIndepVars.size()+1){
        mBandwidthInitilize = bandwidthInitilize;
    }
    else{

    }
    
}

inline vector<double> CGwmMGWR::bandwidthSelectThreshold() const
{
    return mBandwidthSelectThreshold;
}

inline void CGwmMGWR::setBandwidthSelectThreshold(const vector<double> &bandwidthSelectThreshold)
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
        #ifdef ENABLE_OPENMP
        | ParallelType::OpenMP
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
/*
inline GwmBasicGWRAlgorithm::OLSVar CGwmMGWR::getOLSVar() const
{
    return mOLSVar;
}
*/
inline void CGwmMGWR::setOLS(bool value)
{
    mOLS = value;
}


#endif // GWMMULTISCALEGWRTASKTHREAD_H
