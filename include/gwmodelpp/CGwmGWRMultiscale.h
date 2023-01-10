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


namespace gwm
{

class CGwmGWRMultiscale : public CGwmSpatialMultiscaleAlgorithm, public IGwmBandwidthSelectable,public IGwmOpenmpParallelizable,public IGwmRegressionAnalysis 
{
public:
    enum BandwidthInitilizeType
    {
        Null,
        Initial,
        Specified
    };
    static std::unordered_map<BandwidthInitilizeType,std::string> BandwidthInitilizeTypeNameMapper;

    enum BandwidthSelectionCriterionType
    {
        CV,
        AIC
    };
    static std::unordered_map<BandwidthSelectionCriterionType,std::string> BandwidthSelectionCriterionTypeNameMapper;

    enum BackFittingCriterionType
    {
        CVR,
        dCVR
    };
    static std::unordered_map<BackFittingCriterionType,std::string> BackFittingCriterionTypeNameMapper;

    typedef double (CGwmGWRMultiscale::*BandwidthSizeCriterionFunction)(CGwmBandwidthWeight*);
    typedef arma::mat (CGwmGWRMultiscale::*FitAllFunction)(const arma::mat&, const arma::vec&);
    typedef arma::vec (CGwmGWRMultiscale::*FitVarFunction)(const arma::vec&, const arma::vec&, const arma::uword, arma::mat&);
    typedef arma::mat (CGwmGWRMultiscale::*FitCalculator)(const arma::mat&, const arma::vec&, arma::mat&, arma::vec&, arma::vec&, arma::mat&);

private:
    static arma::vec Fitted(const arma::mat& x, const arma::mat& betas)
    {
        return sum(betas % x, 1);
    }

    static double RSS(const arma::mat& x, const arma::mat& y, const arma::mat& betas)
    {
        arma::vec r = y - Fitted(x, betas);
        return sum(r % r);
    }

    static double AICc(const arma::mat& x, const arma::mat& y, const arma::mat& betas, const arma::vec& shat)
    {
        double ss = RSS(x, y, betas), n = (double)x.n_rows;
        return n * log(ss / n) + n * log(2 * arma::datum::pi) + n * ((n + shat(0)) / (n - 2 - shat(0)));
    }
    bool isStoreS()
    {
        return mHasHatMatrix && (mCoords.n_rows < 8192);
    }

    static GwmRegressionDiagnostic CalcDiagnostic(const arma::mat &x, const arma::vec &y, const arma::vec &shat, double RSS);

    void setInitSpatialWeight(const CGwmSpatialWeight &spatialWeight)
    {
        mInitSpatialWeight = spatialWeight;
    }

public:
    CGwmGWRMultiscale() {}

    CGwmGWRMultiscale(const arma::mat& x, const arma::vec& y, const arma::mat& coords, const std::vector<CGwmSpatialWeight>& spatialWeights)
        : CGwmSpatialMultiscaleAlgorithm(coords, spatialWeights)
    {
        mX = x;
        mY = y;
        if (spatialWeights.size() > 0)
            setInitSpatialWeight(spatialWeights[0]);
    }

    virtual ~CGwmGWRMultiscale() {}

public:

    std::vector<BandwidthInitilizeType> bandwidthInitilize() const { return CGwmGWRMultiscale::mBandwidthInitilize; }
    void setBandwidthInitilize(const std::vector<BandwidthInitilizeType> &bandwidthInitilize);

    std::vector<BandwidthSelectionCriterionType> bandwidthSelectionApproach() const { return CGwmGWRMultiscale::mBandwidthSelectionApproach; }
    void setBandwidthSelectionApproach(const std::vector<BandwidthSelectionCriterionType> &bandwidthSelectionApproach);

    std::vector<bool> preditorCentered() const { return mPreditorCentered; }
    void setPreditorCentered(const std::vector<bool> &preditorCentered) { mPreditorCentered = preditorCentered; }

    std::vector<double> bandwidthSelectThreshold() const { return mBandwidthSelectThreshold; }
    void setBandwidthSelectThreshold(const std::vector<double> &bandwidthSelectThreshold) { mBandwidthSelectThreshold = bandwidthSelectThreshold; }

    bool hasHatMatrix() const { return mHasHatMatrix; }
    void setHasHatMatrix(bool hasHatMatrix) { mHasHatMatrix = hasHatMatrix; }

    size_t bandwidthSelectRetryTimes() const { return (size_t)mBandwidthSelectRetryTimes; }
    void setBandwidthSelectRetryTimes(size_t bandwidthSelectRetryTimes) { mBandwidthSelectRetryTimes = (arma::uword)bandwidthSelectRetryTimes; }

    size_t maxIteration() const { return mMaxIteration; }
    void setMaxIteration(size_t maxIteration) { mMaxIteration = maxIteration; }

    BackFittingCriterionType criterionType() const { return mCriterionType; }
    void setCriterionType(const BackFittingCriterionType &criterionType) { mCriterionType = criterionType; }

    double criterionThreshold() const { return mCriterionThreshold; }
    void setCriterionThreshold(double criterionThreshold) { mCriterionThreshold = criterionThreshold; }

    int adaptiveLower() const { return mAdaptiveLower; }
    void setAdaptiveLower(int adaptiveLower) { mAdaptiveLower = adaptiveLower; }

    arma::mat betas() const { return mBetas; }

    BandwidthSizeCriterionFunction bandwidthSizeCriterionAll(BandwidthSelectionCriterionType type);
    BandwidthSizeCriterionFunction bandwidthSizeCriterionVar(BandwidthSelectionCriterionType type);


public:     // GwmTaskThread interface
    std::string name() const { return "Multiscale GWR"; }//override 


public:     // GwmSpatialAlgorithm interface
    bool isValid() override;


public:     // GwmSpatialMultiscaleAlgorithm interface
    virtual void setSpatialWeights(const std::vector<CGwmSpatialWeight> &spatialWeights) override;


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

    arma::mat predict(const arma::mat& locations) override { return arma::mat(locations.n_rows, mX.n_cols, arma::fill::zeros); }

    arma::mat fit() override;

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

    arma::mat fitAllSerial(const arma::mat& x, const arma::vec& y);

    arma::mat fitAllOmp(const arma::mat& x, const arma::vec& y);

    arma::vec fitVarSerial(const arma::vec& x, const arma::vec& y, const arma::uword var, arma::mat& S);

    arma::vec fitVarOmp(const arma::vec& x, const arma::vec& y, const arma::uword var, arma::mat& S);

    arma::mat backfitting(const arma::mat &x, const arma::vec &y);

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
    FitAllFunction mFitAll = &CGwmGWRMultiscale::fitAllSerial;
    FitVarFunction mFitVar = &CGwmGWRMultiscale::fitVarSerial;

    CGwmSpatialWeight mInitSpatialWeight;
    BandwidthSizeCriterionFunction mBandwidthSizeCriterion = &CGwmGWRMultiscale::bandwidthSizeCriterionAllCVSerial;
    size_t mBandwidthSelectionCurrentIndex = 0;


    std::vector<BandwidthInitilizeType> mBandwidthInitilize;
    std::vector<BandwidthSelectionCriterionType> mBandwidthSelectionApproach;
    std::vector<bool> mPreditorCentered;
    std::vector<double> mBandwidthSelectThreshold;
    arma::uword mBandwidthSelectRetryTimes = 5;
    size_t mMaxIteration = 500;
    BackFittingCriterionType mCriterionType = BackFittingCriterionType::dCVR;
    double mCriterionThreshold = 1e-6;
    int mAdaptiveLower = 10;

    bool mHasHatMatrix = true;

    arma::mat mX;
    arma::vec mY;
    arma::mat mBetas;
    arma::mat mBetasSE;
    arma::mat mBetasTV;
    bool mHasIntercept = true;

    arma::mat mS0;
    arma::cube mSArray;
    arma::cube mC;
    arma::mat mX0;
    arma::vec mY0;
    arma::vec mXi;
    arma::vec mYi;

    double mRSS0;

    GwmRegressionDiagnostic mDiagnostic;

    ParallelType mParallelType = ParallelType::SerialOnly;
    int mOmpThreadNum = 8;

public:
    static int treeChildCount;
};

}

#endif // GWMMULTISCALEGWRTASKTHREAD_H
