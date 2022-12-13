#ifndef CGWMGWDR_H
#define CGWMGWDR_H

#include <vector>
#include <armadillo>
#include <gsl/gsl_vector.h>
#include "CGwmSpatialAlgorithm.h"
#include "spatialweight/CGwmSpatialWeight.h"
#include "IGwmRegressionAnalysis.h"
#include "CGwmVariableForwardSelector.h"
#include "IGwmParallelizable.h"

using namespace arma;


class CGwmGWDR : public CGwmSpatialAlgorithm, public IGwmRegressionAnalysis, public IGwmVarialbeSelectable, public IGwmOpenmpParallelizable
{
public:
    typedef mat (CGwmGWDR::*PredictCalculator)(const mat&, const mat&, const vec&);

    typedef mat (CGwmGWDR::*FitCalculator)(const mat&, const vec&, mat&, vec&, vec&, mat&);

    enum BandwidthCriterionType
    {
        CV,
        AIC
    };

    typedef double (CGwmGWDR::*BandwidthCriterionCalculator)(const vector<CGwmBandwidthWeight*>&);

    typedef double (CGwmGWDR::*IndepVarCriterionCalculator)(const vector<size_t>&);

public:
    static GwmRegressionDiagnostic CalcDiagnostic(const mat& x, const vec& y, const mat& betas, const vec& shat);

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

public:
    CGwmGWDR() {}

    CGwmGWDR(const mat& x, const vec& y, const mat& coords, const vector<CGwmSpatialWeight>& spatialWeights, bool hasHatMatrix = true, bool hasIntercept = true)
        : CGwmSpatialAlgorithm(coords)
    {
        mX = x;
        mY = y;
        mSpatialWeights = spatialWeights;
        mHasHatMatrix = hasHatMatrix;
        mHasIntercept = hasIntercept;
    }

    virtual ~CGwmGWDR() {}

public:
    mat betas() const { return mBetas; }

    bool hasHatMatrix() const { return mHasHatMatrix; }

    void setHasHatMatrix(bool flag) { mHasHatMatrix = flag; }

    vector<CGwmSpatialWeight> spatialWeights() { return mSpatialWeights; }

    void setSpatialWeights(vector<CGwmSpatialWeight> spatialWeights) { mSpatialWeights = spatialWeights; }

    bool enableBandwidthOptimize() { return mEnableBandwidthOptimize; }

    void setEnableBandwidthOptimize(bool flag) { mEnableBandwidthOptimize = flag; }

    double bandwidthOptimizeEps() const { return mBandwidthOptimizeEps; }

    void setBandwidthOptimizeEps(double value) { mBandwidthOptimizeEps = value; }

    size_t bandwidthOptimizeMaxIter() const { return mBandwidthOptimizeMaxIter; }

    void setBandwidthOptimizeMaxIter(size_t value) { mBandwidthOptimizeMaxIter = value; }

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

public: // CGwmAlgorithm
    bool isValid() override;

public: // IGwmRegressionAnalysis
    virtual arma::vec dependentVariable() const override { return mY; }
    virtual void setDependentVariable(const arma::vec& y) override { mY = y; }

    virtual arma::mat independentVariables() const override { return mX; }
    virtual void setIndependentVariables(const arma::mat& x) override { mX = x; }

    virtual bool hasIntercept() const override { return mHasIntercept; }
    virtual void setHasIntercept(const bool has) override { mHasIntercept = has; }

    virtual GwmRegressionDiagnostic diagnostic() const override { return mDiagnostic; }

    virtual mat predict(const mat& locations) override { return mat(locations.n_rows, mX.n_cols, arma::fill::zeros); }

    virtual mat fit() override;

public:  // IGwmVariableSelectable
    double getCriterion(const vector<size_t>& variables) override
    {
        return (this->*mIndepVarCriterionFunction)(variables);
    }

    std::vector<std::size_t> selectedVariables() override
    {
        return mSelectedIndepVars;
    }

public:  // IGwmOpenmpParallelizable
    int parallelAbility() const
    {
        return ParallelType::SerialOnly
        #ifdef ENABLE_OPENMP
        | ParallelType::OpenMP
        #endif        
        ;
    }
    
    ParallelType parallelType() const
    {
        return mParallelType;
    }
    
    void setParallelType(const ParallelType& type);

    void setOmpThreadNum(const int threadNum)
    {
        mOmpThreadNum = threadNum;
    }


public:
    double bandwidthCriterion(const vector<CGwmBandwidthWeight*>& bandwidths)
    {
        return (this->*mBandwidthCriterionFunction)(bandwidths);
    }

protected:
    mat predictSerial(const mat& locations, const mat& x, const vec& y);
    mat predictOmp(const mat& locations, const mat& x, const vec& y);
    mat fitSerial(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qdiag, mat& S);
    mat fitOmp(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qdiag, mat& S);

    double bandwidthCriterionAICSerial(const vector<CGwmBandwidthWeight*>& bandwidths);
    double bandwidthCriterionCVSerial(const vector<CGwmBandwidthWeight*>& bandwidths);
    double indepVarCriterionSerial(const vector<size_t>& indepVars);

#ifdef ENABLE_OPENMP
    double bandwidthCriterionAICOmp(const vector<CGwmBandwidthWeight*>& bandwidths);
    double bandwidthCriterionCVOmp(const vector<CGwmBandwidthWeight*>& bandwidths);
    double indepVarCriterionOmp(const vector<size_t>& indepVars);
#endif

private:
    bool isStoreS()
    {
        return mHasHatMatrix && (mCoords.n_rows < 8192);
    }

private:
    vector<CGwmSpatialWeight> mSpatialWeights;

    vec mY;
    mat mX;
    mat mBetas;
    bool mHasIntercept = true;
    bool mHasHatMatrix = true;
    GwmRegressionDiagnostic mDiagnostic;

    PredictCalculator mPredictFunction = &CGwmGWDR::predictSerial;
    FitCalculator mFitFunction = &CGwmGWDR::fitSerial;

    bool mEnableBandwidthOptimize = false;
    BandwidthCriterionType mBandwidthCriterionType = BandwidthCriterionType::CV;
    BandwidthCriterionCalculator mBandwidthCriterionFunction = &CGwmGWDR::bandwidthCriterionCVSerial;
    double mBandwidthOptimizeEps = 1e-6;
    size_t mBandwidthOptimizeMaxIter = 100000;
    double mBandwidthOptimizeStep = 0.01;

    bool mEnableIndepVarSelect = false;
    double mIndepVarSelectThreshold = 3.0;
    VariablesCriterionList mIndepVarCriterionList;
    IndepVarCriterionCalculator mIndepVarCriterionFunction = &CGwmGWDR::indepVarCriterionSerial;
    std::vector<std::size_t> mSelectedIndepVars;

    ParallelType mParallelType = ParallelType::SerialOnly;
    int mOmpThreadNum = 8;

    arma::mat mBetasSE;
    arma::vec mSHat;
    arma::vec mQDiag;
    arma::mat mS;
};


class CGwmGWDRBandwidthOptimizer
{
public:
    struct Parameter
    {
        CGwmGWDR* instance;
        vector<CGwmBandwidthWeight*>* bandwidths;
        uword featureCount;
    };

    static double criterion_function(const gsl_vector* bws, void* params);

public:
    CGwmGWDRBandwidthOptimizer(vector<CGwmBandwidthWeight*> weights)
    {
        mBandwidths = weights;
    }

    const int optimize(CGwmGWDR* instance, uword featureCount, size_t maxIter, double eps, double step);

private:
    vector<CGwmBandwidthWeight*> mBandwidths;
    Parameter mParameter;
};

#endif  // CGWMGWDR_H