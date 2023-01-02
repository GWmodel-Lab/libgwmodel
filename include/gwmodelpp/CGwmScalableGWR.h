#ifndef CGWMScalableGWR_H
#define CGWMScalableGWR_H

#include <utility>
#include <string>
#include <initializer_list>
#include "CGwmGWRBase.h"
#include "GwmRegressionDiagnostic.h"
#include "IGwmBandwidthSelectable.h"
#include "IGwmVarialbeSelectable.h"
#include "IGwmParallelizable.h"
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>

using namespace std;

class CGwmScalableGWR : public CGwmGWRBase
{
public:
    enum BandwidthSelectionCriterionType
    {
        AIC,
        CV
    };

    enum NameFormat
    {
        Fixed,
        VarName,
        PrefixVarName,
        SuffixVariable
    };

    struct LoocvParams
    {
        const mat* x;
        const mat* y;
        const uword polynomial;
        const mat* Mx0;
        const mat* My0;
    };

    typedef tuple<string, mat, NameFormat> ResultLayerDataItem;

    static double Loocv(const vec& target, const mat& x, const vec& y, uword poly, const mat& Mx0, const mat& My0);
    static double AICvalue(const vec& target, const mat& x, const vec& y, uword poly, const mat& Mx0, const mat& My0);

public:
    static size_t treeChildCount;

private:
    static GwmRegressionDiagnostic CalcDiagnostic(const mat& x, const vec& y, const mat& betas, const vec& shat);

public:
    CGwmScalableGWR(){};
    ~CGwmScalableGWR(){};

    uword polynomial() const { return mPolynomial; }

    void setPolynomial(uword polynomial) { mPolynomial = polynomial; }

    double cv() const { return mCV; }

    double scale() const { return mScale; }

    double penalty() const { return mPenalty; }

    bool hasHatMatrix() const { return mHasHatMatrix; }

    void setHasHatMatrix(const bool has) { mHasHatMatrix = has; }

    BandwidthSelectionCriterionType parameterOptimizeCriterion() const
    {
        return mParameterOptimizeCriterion;
    }

    void setParameterOptimizeCriterion(const BandwidthSelectionCriterionType &parameterOptimizeCriterion)
    {
        mParameterOptimizeCriterion = parameterOptimizeCriterion;
    }

public:     // GwmSpatialAlgorithm interface
    bool isValid() override;


public:     // IRegressionAnalysis interface
    mat fit() override;

    mat predict(const mat& locations) override;

private:
    void findDataPointNeighbours();
    mat findNeighbours(const mat& points, umat &nnIndex);
    double optimize(const mat& Mx0, const mat& My0, double& b_tilde, double& alpha);
    void prepare();

    mat fitSerial(const arma::mat &x, const arma::vec &y);
    mat predictSerial(const mat& locations, const arma::mat& x, const arma::vec& y);

private:
    uword mPolynomial = 4;
    size_t mMaxIter = 500;
    double mCV = 0.0;
    double mScale = 1.0;
    double mPenalty = 0.01;

    bool hasRegressionLayerXY = false;
    vec mRegressionLayerY;
    mat mRegressionLayerX;
    
    bool mHasHatMatrix = true;

    CGwmSpatialWeight mDpSpatialWeight;
    //DistanceParameter* mRegressionDistanceParameter = nullptr;
    //DistanceParameter* mPredictionDistanceParameter = nullptr;

    BandwidthSelectionCriterionType mParameterOptimizeCriterion = BandwidthSelectionCriterionType::CV;
    

    mat mG0;
    umat mDpNNIndex;
    mat mDpNNDists;
    mat mMx0;
    mat mMxx0;
    mat mMy0;
    vec mShat;
    mat mBetasSE;
};

#endif  // CGWMScalableGWR_H
