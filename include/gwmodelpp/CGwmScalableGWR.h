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
        const int bw;
        const int polynomial;
        const mat* Mx0;
        const mat* My0;
    };

    typedef tuple<string, mat, NameFormat> ResultLayerDataItem;

    static double Loocv(const vec& target, const mat& x, const vec& y, int bw, int poly, const mat& Mx0, const mat& My0);
    static double AICvalue(const vec& target, const mat& x, const vec& y, int bw, int poly, const mat& Mx0, const mat& My0);

private:
    static GwmRegressionDiagnostic CalcDiagnostic(const mat& x, const vec& y, const mat& betas, const vec& shat);

public:
    CGwmScalableGWR(){};
    ~CGwmScalableGWR(){};

    //void run() override;
    mat fit();

    int polynomial() const;
    void setPolynomial(int polynomial);

    double cv() const;
    double scale() const;
    double penalty() const;

    mat predictData() const;
    void setPredictData(const mat &locations);

    bool hasHatMatrix() const { return mHasHatMatrix; }

    void setHasHatMatrix(const bool has) { mHasHatMatrix = has; }

    bool hasPredict() const;
    void setHasPredict(bool hasPredict);

    BandwidthSelectionCriterionType parameterOptimizeCriterion() const;
    void setParameterOptimizeCriterion(const BandwidthSelectionCriterionType &parameterOptimizeCriterion);

public:     // GwmSpatialAlgorithm interface
    bool isValid() override;


public:     // IRegressionAnalysis interface
    mat predict(const mat& locations) override
    {
        return fitSerial(mX,mY);
    }
    /*mat predict(const mat &x, const vec &y) override
    {
        return fit(x, y);
    }*/

protected:
    bool hasPredictLayer()
    {
        return mHasPredict;
    }

private:
    void findDataPointNeighbours();
    mat findNeighbours(const CGwmSpatialWeight& spatialWeight, umat &nnIndex);
    double optimize(const mat& Mx0, const mat& My0, double& b_tilde, double& alpha);
    void prepare();

    mat predictSerial(const arma::mat& x, const arma::vec& y);
    mat fitSerial(const arma::mat &x, const arma::vec &y);

    //void createResultLayer(initializer_list<ResultLayerDataItem> items);

    bool mHasPredict = false;

private:
    int mPolynomial = 4;
    int mMaxIter = 500;
    double mCV = 0.0;
    double mScale = 1.0;
    double mPenalty = 0.01;

    bool hasRegressionLayerXY = false;
    vec mRegressionLayerY;
    mat mRegressionLayerX;
    
    mat mPredictData;

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
public:
    static int treeChildCount;
};


inline mat CGwmScalableGWR::predictData() const
{
    return mPredictData;
}

inline void CGwmScalableGWR::setPredictData(const mat &locations)
{
    mPredictData = locations;
}
inline void CGwmScalableGWR::setPolynomial(int polynomial)
{
    mPolynomial = polynomial;
}

inline double CGwmScalableGWR::penalty() const
{
    return mPenalty;
}

inline double CGwmScalableGWR::scale() const
{
    return mScale;
}

inline double CGwmScalableGWR::cv() const
{
    return mCV;
}

inline int CGwmScalableGWR::polynomial() const
{
    return mPolynomial;
}

inline CGwmScalableGWR::BandwidthSelectionCriterionType CGwmScalableGWR::parameterOptimizeCriterion() const
{
    return mParameterOptimizeCriterion;
}

inline void CGwmScalableGWR::setParameterOptimizeCriterion(const BandwidthSelectionCriterionType &parameterOptimizeCriterion)
{
    mParameterOptimizeCriterion = parameterOptimizeCriterion;
}

#endif  // CGWMScalableGWR_H
