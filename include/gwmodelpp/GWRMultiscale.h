#ifndef GWRMULTISCALE_H
#define GWRMULTISCALE_H

#include <utility>
#include <string>
#include <initializer_list>
#include "SpatialMultiscaleAlgorithm.h"
#include "spatialweight/SpatialWeight.h"
#include "IRegressionAnalysis.h"
#include "IBandwidthSelectable.h"
#include "BandwidthSelector.h"
#include "IParallelizable.h"


namespace gwm
{

class GWRMultiscale : public SpatialMultiscaleAlgorithm, public IBandwidthSelectable,public IOpenmpParallelizable,public IRegressionAnalysis 
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

    typedef double (GWRMultiscale::*BandwidthSizeCriterionFunction)(BandwidthWeight*);
    typedef arma::mat (GWRMultiscale::*FitAllFunction)(const arma::mat&, const arma::vec&);
    typedef arma::vec (GWRMultiscale::*FitVarFunction)(const arma::vec&, const arma::vec&, const arma::uword, arma::mat&);
    typedef arma::mat (GWRMultiscale::*FitCalculator)(const arma::mat&, const arma::vec&, arma::mat&, arma::vec&, arma::vec&, arma::mat&);

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

    static RegressionDiagnostic CalcDiagnostic(const arma::mat &x, const arma::vec &y, const arma::vec &shat, double RSS);

    void setInitSpatialWeight(const SpatialWeight &spatialWeight)
    {
        mInitSpatialWeight = spatialWeight;
    }

public:
    GWRMultiscale() {}

    GWRMultiscale(const arma::mat& x, const arma::vec& y, const arma::mat& coords, const std::vector<SpatialWeight>& spatialWeights)
        : SpatialMultiscaleAlgorithm(coords, spatialWeights)
    {
        mX = x;
        mY = y;
        if (spatialWeights.size() > 0)
            setInitSpatialWeight(spatialWeights[0]);
    }

    virtual ~GWRMultiscale() {}

public:

    std::vector<BandwidthInitilizeType> bandwidthInitilize() const { return GWRMultiscale::mBandwidthInitilize; }
    void setBandwidthInitilize(const std::vector<BandwidthInitilizeType> &bandwidthInitilize);

    std::vector<BandwidthSelectionCriterionType> bandwidthSelectionApproach() const { return GWRMultiscale::mBandwidthSelectionApproach; }
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


public:     // TaskThread interface
    std::string name() const { return "Multiscale GWR"; }//override 


public:     // SpatialAlgorithm interface
    bool isValid() override;


public:     // SpatialMultiscaleAlgorithm interface
    virtual void setSpatialWeights(const std::vector<SpatialWeight> &spatialWeights) override;


public:     // IBandwidthSizeSelectable interface
    double getCriterion(BandwidthWeight* weight) override
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

    virtual RegressionDiagnostic diagnostic() const override { return mDiagnostic; }

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

    BandwidthWeight* bandwidth(size_t i)
    {
        return static_cast<BandwidthWeight*>(mSpatialWeights[i].weight());
    }

    arma::mat fitAllSerial(const arma::mat& x, const arma::vec& y);

    arma::mat fitAllOmp(const arma::mat& x, const arma::vec& y);

    arma::vec fitVarSerial(const arma::vec& x, const arma::vec& y, const arma::uword var, arma::mat& S);

    arma::vec fitVarOmp(const arma::vec& x, const arma::vec& y, const arma::uword var, arma::mat& S);

    arma::mat backfitting(const arma::mat &x, const arma::vec &y);

    double bandwidthSizeCriterionAllCVSerial(BandwidthWeight* bandwidthWeight);

    double bandwidthSizeCriterionAllAICSerial(BandwidthWeight* bandwidthWeight);

    double bandwidthSizeCriterionVarCVSerial(BandwidthWeight* bandwidthWeight);

    double bandwidthSizeCriterionVarAICSerial(BandwidthWeight* bandwidthWeight);

#ifdef ENABLE_OPENMP
    double bandwidthSizeCriterionAllCVOmp(BandwidthWeight* bandwidthWeight);

    double bandwidthSizeCriterionAllAICOmp(BandwidthWeight* bandwidthWeight);

    double bandwidthSizeCriterionVarCVOmp(BandwidthWeight* bandwidthWeight);

    double bandwidthSizeCriterionVarAICOmp(BandwidthWeight* bandwidthWeight);
#endif

    void createInitialDistanceParameter();

private:
    FitAllFunction mFitAll = &GWRMultiscale::fitAllSerial;
    FitVarFunction mFitVar = &GWRMultiscale::fitVarSerial;

    SpatialWeight mInitSpatialWeight;
    BandwidthSizeCriterionFunction mBandwidthSizeCriterion = &GWRMultiscale::bandwidthSizeCriterionAllCVSerial;
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

    RegressionDiagnostic mDiagnostic;

    ParallelType mParallelType = ParallelType::SerialOnly;
    int mOmpThreadNum = 8;

public:
    static int treeChildCount;
};

}

#endif // GWRMULTISCALE_H
