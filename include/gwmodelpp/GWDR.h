#ifndef GWDR_H
#define GWDR_H

#include <vector>
#include <armadillo>
#include <gsl/gsl_vector.h>
#include "SpatialAlgorithm.h"
#include "spatialweight/SpatialWeight.h"
#include "IRegressionAnalysis.h"
#include "VariableForwardSelector.h"
#include "IParallelizable.h"

namespace gwm
{

class GWDR : public SpatialAlgorithm, public IRegressionAnalysis, public IVarialbeSelectable, public IOpenmpParallelizable
{
public:
    typedef arma::mat (GWDR::*PredictCalculator)(const arma::mat&, const arma::mat&, const arma::vec&);

    typedef arma::mat (GWDR::*FitCalculator)(const arma::mat&, const arma::vec&, arma::mat&, arma::vec&, arma::vec&, arma::mat&);

    enum BandwidthCriterionType
    {
        CV,
        AIC
    };

    typedef double (GWDR::*BandwidthCriterionCalculator)(const std::vector<BandwidthWeight*>&);

    typedef double (GWDR::*IndepVarCriterionCalculator)(const std::vector<std::size_t>&);

public:
    static RegressionDiagnostic CalcDiagnostic(const arma::mat& x, const arma::vec& y, const arma::mat& betas, const arma::vec& shat);

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

public:
    GWDR() {}

    GWDR(const arma::mat& x, const arma::vec& y, const arma::mat& coords, const std::vector<SpatialWeight>& spatialWeights, bool hasHatMatrix = true, bool hasIntercept = true)
        : SpatialAlgorithm(coords)
    {
        mX = x;
        mY = y;
        mSpatialWeights = spatialWeights;
        mHasHatMatrix = hasHatMatrix;
        mHasIntercept = hasIntercept;
    }

    virtual ~GWDR() {}

public:
    arma::mat betas() const { return mBetas; }

    bool hasHatMatrix() const { return mHasHatMatrix; }

    void setHasHatMatrix(bool flag) { mHasHatMatrix = flag; }

    std::vector<SpatialWeight> spatialWeights() { return mSpatialWeights; }

    void setSpatialWeights(std::vector<SpatialWeight> spatialWeights) { mSpatialWeights = spatialWeights; }

    bool enableBandwidthOptimize() { return mEnableBandwidthOptimize; }

    void setEnableBandwidthOptimize(bool flag) { mEnableBandwidthOptimize = flag; }

    double bandwidthOptimizeEps() const { return mBandwidthOptimizeEps; }

    void setBandwidthOptimizeEps(double value) { mBandwidthOptimizeEps = value; }

    std::size_t bandwidthOptimizeMaxIter() const { return mBandwidthOptimizeMaxIter; }

    void setBandwidthOptimizeMaxIter(std::size_t value) { mBandwidthOptimizeMaxIter = value; }

    double bandwidthOptimizeStep() const { return mBandwidthOptimizeStep; }

    void setBandwidthOptimizeStep(double value) { mBandwidthOptimizeStep = value; }

    BandwidthCriterionType bandwidthCriterionType() const { return mBandwidthCriterionType; }

    void setBandwidthCriterionType(const BandwidthCriterionType& type);

    bool enableIndpenVarSelect() const { return mEnableIndepVarSelect; }

    void setEnableIndepVarSelect(bool flag) { mEnableIndepVarSelect = flag; }

    VariablesCriterionList indepVarCriterionList() const { return mIndepVarCriterionList; }

    arma::mat betasSE() { return mBetasSE; }

    arma::vec sHat() { return mSHat; }

    arma::vec qDiag() { return mQDiag; }

    arma::mat s() { return mS; }

public: // Algorithm
    bool isValid() override;

public: // IRegressionAnalysis
    virtual arma::vec dependentVariable() const override { return mY; }
    virtual void setDependentVariable(const arma::vec& y) override { mY = y; }

    virtual arma::mat independentVariables() const override { return mX; }
    virtual void setIndependentVariables(const arma::mat& x) override { mX = x; }

    virtual bool hasIntercept() const override { return mHasIntercept; }
    virtual void setHasIntercept(const bool has) override { mHasIntercept = has; }

    virtual RegressionDiagnostic diagnostic() const override { return mDiagnostic; }

    virtual arma::mat predict(const arma::mat& locations) override { return arma::mat(locations.n_rows, mX.n_cols, arma::fill::zeros); }

    virtual arma::mat fit() override;

public:  // IVariableSelectable
    double getCriterion(const std::vector<std::size_t>& variables) override
    {
        return (this->*mIndepVarCriterionFunction)(variables);
    }

    std::vector<std::size_t> selectedVariables() override
    {
        return mSelectedIndepVars;
    }

public:  // IOpenmpParallelizable
    int parallelAbility() const override
    {
        return ParallelType::SerialOnly
        #ifdef ENABLE_OPENMP
        | ParallelType::OpenMP
        #endif        
        ;
    }
    
    ParallelType parallelType() const override
    {
        return mParallelType;
    }
    
    void setParallelType(const ParallelType& type) override;

    void setOmpThreadNum(const int threadNum) override
    {
        mOmpThreadNum = threadNum;
    }


public:
    double bandwidthCriterion(const std::vector<BandwidthWeight*>& bandwidths)
    {
        return (this->*mBandwidthCriterionFunction)(bandwidths);
    }

protected:
    arma::mat predictSerial(const arma::mat& locations, const arma::mat& x, const arma::vec& y);
    arma::mat predictOmp(const arma::mat& locations, const arma::mat& x, const arma::vec& y);
    arma::mat fitSerial(const arma::mat& x, const arma::vec& y, arma::mat& betasSE, arma::vec& shat, arma::vec& qdiag, arma::mat& S);
    arma::mat fitOmp(const arma::mat& x, const arma::vec& y, arma::mat& betasSE, arma::vec& shat, arma::vec& qdiag, arma::mat& S);

    double bandwidthCriterionAICSerial(const std::vector<BandwidthWeight*>& bandwidths);
    double bandwidthCriterionCVSerial(const std::vector<BandwidthWeight*>& bandwidths);
    double indepVarCriterionSerial(const std::vector<std::size_t>& indepVars);

#ifdef ENABLE_OPENMP
    double bandwidthCriterionAICOmp(const std::vector<BandwidthWeight*>& bandwidths);
    double bandwidthCriterionCVOmp(const std::vector<BandwidthWeight*>& bandwidths);
    double indepVarCriterionOmp(const std::vector<std::size_t>& indepVars);
#endif

private:
    bool isStoreS()
    {
        return mHasHatMatrix && (mCoords.n_rows < 8192);
    }

private:
    std::vector<SpatialWeight> mSpatialWeights;

    arma::vec mY;
    arma::mat mX;
    arma::mat mBetas;
    bool mHasIntercept = true;
    bool mHasHatMatrix = true;
    RegressionDiagnostic mDiagnostic;

    PredictCalculator mPredictFunction = &GWDR::predictSerial;
    FitCalculator mFitFunction = &GWDR::fitSerial;

    bool mEnableBandwidthOptimize = false;
    BandwidthCriterionType mBandwidthCriterionType = BandwidthCriterionType::CV;
    BandwidthCriterionCalculator mBandwidthCriterionFunction = &GWDR::bandwidthCriterionCVSerial;
    double mBandwidthOptimizeEps = 1e-6;
    std::size_t mBandwidthOptimizeMaxIter = 100000;
    double mBandwidthOptimizeStep = 0.01;

    bool mEnableIndepVarSelect = false;
    double mIndepVarSelectThreshold = 3.0;
    VariablesCriterionList mIndepVarCriterionList;
    IndepVarCriterionCalculator mIndepVarCriterionFunction = &GWDR::indepVarCriterionSerial;
    std::vector<std::size_t> mSelectedIndepVars;

    ParallelType mParallelType = ParallelType::SerialOnly;
    int mOmpThreadNum = 8;

    arma::mat mBetasSE;
    arma::vec mSHat;
    arma::vec mQDiag;
    arma::mat mS;
};


class GWDRBandwidthOptimizer
{
public:
    struct Parameter
    {
        GWDR* instance;
        std::vector<BandwidthWeight*>* bandwidths;
        arma::uword featureCount;
    };

    static double criterion_function(const gsl_vector* bws, void* params);

public:
    GWDRBandwidthOptimizer(std::vector<BandwidthWeight*> weights)
    {
        mBandwidths = weights;
    }

    const int optimize(GWDR* instance, arma::uword featureCount, std::size_t maxIter, double eps, double step);

private:
    std::vector<BandwidthWeight*> mBandwidths;
};

}

#endif  // GWDR_H