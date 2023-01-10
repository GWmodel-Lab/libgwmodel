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


class CGwmGWRScalable : public CGwmGWRBase
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
        const arma::mat* x;
        const arma::mat* y;
        const arma::uword polynomial;
        const arma::mat* Mx0;
        const arma::mat* My0;
    };

    typedef std::tuple<std::string, arma::mat, NameFormat> ResultLayerDataItem;

    static double Loocv(const arma::vec& target, const arma::mat& x, const arma::vec& y, arma::uword poly, const arma::mat& Mx0, const arma::mat& My0);
    static double AICvalue(const arma::vec& target, const arma::mat& x, const arma::vec& y, arma::uword poly, const arma::mat& Mx0, const arma::mat& My0);

public:
    static size_t treeChildCount;

private:
    static GwmRegressionDiagnostic CalcDiagnostic(const arma::mat& x, const arma::vec& y, const arma::mat& betas, const arma::vec& shat);

public:
    CGwmGWRScalable(){};
    ~CGwmGWRScalable(){};

    arma::uword polynomial() const { return mPolynomial; }

    void setPolynomial(arma::uword polynomial) { mPolynomial = polynomial; }

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
    arma::mat fit() override;

    arma::mat predict(const arma::mat& locations) override;

private:
    void findDataPointNeighbours();
    arma::mat findNeighbours(const arma::mat& points, arma::umat &nnIndex);
    double optimize(const arma::mat& Mx0, const arma::mat& My0, double& b_tilde, double& alpha);
    void prepare();

    arma::mat fitSerial(const arma::mat &x, const arma::vec &y);
    arma::mat predictSerial(const arma::mat& locations, const arma::mat& x, const arma::vec& y);

private:
    arma::uword mPolynomial = 4;
    size_t mMaxIter = 500;
    double mCV = 0.0;
    double mScale = 1.0;
    double mPenalty = 0.01;

    bool mHasHatMatrix = true;

    CGwmSpatialWeight mDpSpatialWeight;
    //DistanceParameter* mRegressionDistanceParameter = nullptr;
    //DistanceParameter* mPredictionDistanceParameter = nullptr;

    BandwidthSelectionCriterionType mParameterOptimizeCriterion = BandwidthSelectionCriterionType::CV;
    

    arma::mat mG0;
    arma::umat mDpNNIndex;
    arma::mat mDpNNDists;
    arma::mat mMx0;
    arma::mat mMxx0;
    arma::mat mMy0;
    arma::vec mShat;
    arma::mat mBetasSE;
};

#endif  // CGWMScalableGWR_H
