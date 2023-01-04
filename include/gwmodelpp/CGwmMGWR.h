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

    typedef double (CGwmMGWR::*BandwidthSizeCriterionFunction)(CGwmBandwidthWeight*);
    typedef mat (CGwmMGWR::*FitAllFunction)(const arma::mat&, const arma::vec&);
    typedef vec (CGwmMGWR::*FitVarFunction)(const arma::vec&, const arma::vec&, const arma::uword, mat&);
    typedef mat (CGwmMGWR::*FitCalculator)(const mat&, const vec&, mat&, vec&, vec&, mat&);

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
        double ss = RSS(x, y, betas), n = (double)x.n_rows;
        return n * log(ss / n) + n * log(2 * datum::pi) + n * ((n + shat(0)) / (n - 2 - shat(0)));
    }
    bool isStoreS()
    {
        return mHasHatMatrix && (mCoords.n_rows < 8192);
    }

    static GwmRegressionDiagnostic CalcDiagnostic(const mat &x, const vec &y, const mat &S0, double RSS);

    void setInitSpatialWeight(const CGwmSpatialWeight &spatialWeight)
    {
        mInitSpatialWeight = spatialWeight;
    }

public:
    CGwmMGWR() {}

    CGwmMGWR(const mat& x, const vec& y, const arma::mat& coords, const std::vector<CGwmSpatialWeight>& spatialWeights)
        : CGwmSpatialMultiscaleAlgorithm(coords, spatialWeights)
    {
        mX = x;
        mY = y;
    }

    virtual ~CGwmMGWR() {}

public:

    vector<BandwidthInitilizeType> bandwidthInitilize() const { return CGwmMGWR::mBandwidthInitilize; }
    void setBandwidthInitilize(const vector<BandwidthInitilizeType> &bandwidthInitilize);

    vector<BandwidthSelectionCriterionType> bandwidthSelectionApproach() const { return CGwmMGWR::mBandwidthSelectionApproach; }
    void setBandwidthSelectionApproach(const vector<BandwidthSelectionCriterionType> &bandwidthSelectionApproach);

    vector<bool> preditorCentered() const { return mPreditorCentered; }
    void setPreditorCentered(const vector<bool> &preditorCentered) { mPreditorCentered = preditorCentered; }

    vector<double> bandwidthSelectThreshold() const { return mBandwidthSelectThreshold; }
    void setBandwidthSelectThreshold(const vector<double> &bandwidthSelectThreshold) { mBandwidthSelectThreshold = bandwidthSelectThreshold; }

    bool hasHatMatrix() const { return mHasHatMatrix; }
    void setHasHatMatrix(bool hasHatMatrix) { mHasHatMatrix = hasHatMatrix; }

    size_t bandwidthSelectRetryTimes() const { return (size_t)mBandwidthSelectRetryTimes; }
    void setBandwidthSelectRetryTimes(size_t bandwidthSelectRetryTimes) { mBandwidthSelectRetryTimes = (uword)bandwidthSelectRetryTimes; }

    size_t maxIteration() const { return mMaxIteration; }
    void setMaxIteration(size_t maxIteration) { mMaxIteration = maxIteration; }

    BackFittingCriterionType criterionType() const { return mCriterionType; }
    void setCriterionType(const BackFittingCriterionType &criterionType) { mCriterionType = criterionType; }

    double criterionThreshold() const { return mCriterionThreshold; }
    void setCriterionThreshold(double criterionThreshold) { mCriterionThreshold = criterionThreshold; }

    int adaptiveLower() const { return mAdaptiveLower; }
    void setAdaptiveLower(int adaptiveLower) { mAdaptiveLower = adaptiveLower; }

    mat betas() const { return mBetas; }

    BandwidthSizeCriterionFunction bandwidthSizeCriterionAll(BandwidthSelectionCriterionType type);
    BandwidthSizeCriterionFunction bandwidthSizeCriterionVar(BandwidthSelectionCriterionType type);


public:     // GwmTaskThread interface
    string name() const { return "Multiscale GWR"; }//override 


public:     // GwmSpatialAlgorithm interface
    bool isValid() override;


public:     // GwmSpatialMultiscaleAlgorithm interface
    virtual void setSpatialWeights(const vector<CGwmSpatialWeight> &spatialWeights) override;


public:     // IBandwidthSizeSelectable interface
    double getCriterion(CGwmBandwidthWeight* weight) override
    {
        return (this->*mBandwidthSizeCriterion)(weight);
    }


public:     // IRegressionAnalysis interface
    virtual arma::vec dependentVariable() const override { return mY; }
    virtual void setDependentVariable(const arma::vec& y) override { mY = y; }

    virtual arma::mat independentVariables() const override { return mX; }
    virtual void setIndependentVariables(const arma::mat& x) override { mX = x; }

    virtual bool hasIntercept() const override { return mHasIntercept; }
    virtual void setHasIntercept(const bool has) override { mHasIntercept = has; }

    virtual GwmRegressionDiagnostic diagnostic() const override { return mDiagnostic; }

    mat predict(const mat& locations) override { return mat(locations.n_rows, mX.n_cols, arma::fill::zeros); }

    mat fit() override;

public:     // IParallelalbe interface
    int parallelAbility() const override 
    {
        return ParallelType::SerialOnly
#ifdef ENABLE_OPENMP
            | ParallelType::OpenMP
#endif        
            ;
    }

    ParallelType parallelType() const override { return mParallelType; }

    void setParallelType(const ParallelType &type) override;


public:     // IOpenmpParallelable interface
    void setOmpThreadNum(const int threadNum) override { mOmpThreadNum = threadNum; }


protected:

    CGwmBandwidthWeight* bandwidth(size_t i)
    {
        return static_cast<CGwmBandwidthWeight*>(mSpatialWeights[i].weight());
    }

    mat fitAllSerial(const mat& x, const vec& y);

    mat fitAllOmp(const mat& x, const vec& y);

    vec fitVarSerial(const vec& x, const vec& y, const uword var, mat& S);

    vec fitVarOmp(const vec& x, const vec& y, const uword var, mat& S);

    mat backfitting(const mat &x, const vec &y);

    double bandwidthSizeCriterionAllCVSerial(CGwmBandwidthWeight* bandwidthWeight);

    double bandwidthSizeCriterionAllAICSerial(CGwmBandwidthWeight* bandwidthWeight);

    double bandwidthSizeCriterionVarCVSerial(CGwmBandwidthWeight* bandwidthWeight);

    double bandwidthSizeCriterionVarAICSerial(CGwmBandwidthWeight* bandwidthWeight);

#ifdef ENABLE_OPENMP
    double bandwidthSizeCriterionAllCVOmp(CGwmBandwidthWeight* bandwidthWeight);

    double bandwidthSizeCriterionAllAICOmp(CGwmBandwidthWeight* bandwidthWeight);

    double bandwidthSizeCriterionVarCVOmp(CGwmBandwidthWeight* bandwidthWeight);

    double bandwidthSizeCriterionVarAICOmp(CGwmBandwidthWeight* bandwidthWeight);
#endif

    void createInitialDistanceParameter();

private:
    FitAllFunction mFitAll = &CGwmMGWR::fitAllSerial;
    FitVarFunction mFitVar = &CGwmMGWR::fitVarSerial;

    CGwmSpatialWeight mInitSpatialWeight;
    BandwidthSizeCriterionFunction mBandwidthSizeCriterion = &CGwmMGWR::bandwidthSizeCriterionAllCVSerial;
    size_t mBandwidthSelectionCurrentIndex = 0;


    vector<BandwidthInitilizeType> mBandwidthInitilize;
    vector<BandwidthSelectionCriterionType> mBandwidthSelectionApproach;
    vector<bool> mPreditorCentered;
    vector<double> mBandwidthSelectThreshold;
    uword mBandwidthSelectRetryTimes = 5;
    size_t mMaxIteration = 500;
    BackFittingCriterionType mCriterionType = BackFittingCriterionType::dCVR;
    double mCriterionThreshold = 1e-6;
    int mAdaptiveLower = 10;

    bool mHasHatMatrix = true;

    mat mX;
    vec mY;
    mat mBetas;
    mat mBetasSE;
    mat mBetasTV;
    bool mHasIntercept = true;

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

#endif // GWMMULTISCALEGWRTASKTHREAD_H
