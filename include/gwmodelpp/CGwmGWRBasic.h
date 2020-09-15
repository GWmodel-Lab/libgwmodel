#ifndef CGWMGWRBASIC_H
#define CGWMGWRBASIC_H

#include <utility>
#include <string>
#include <initializer_list>
#include "CGwmGWRBase.h"
#include "GwmRegressionDiagnostic.h"
#include "CGwmBandwidthSelector.h"

using namespace std;

class CGwmGWRBasic : public CGwmGWRBase, public IGwmBandwidthSelectable
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

    static unordered_map<BandwidthSelectionCriterionType, string> BandwidthSelectionCriterionTypeNameMapper;
    
    typedef mat (CGwmGWRBasic::*RegressionCalculator)(const mat&, const vec&);
    typedef mat (CGwmGWRBasic::*RegressionHatmatrixCalculator)(const mat&, const vec&, mat&, vec&, vec&, mat&);

    typedef tuple<string, mat, NameFormat> ResultLayerDataItem;

    typedef double (CGwmGWRBasic::*BandwidthSelectionCriterionCalculator)(CGwmBandwidthWeight*);

private:
    static GwmRegressionDiagnostic CalcDiagnostic(const mat& x, const vec& y, const mat& betas, const vec& shat);

public:
    CGwmGWRBasic();
    ~CGwmGWRBasic();

public:
    bool isAutoselectBandwidth() const;
    void setIsAutoselectBandwidth(bool isAutoSelect);

    BandwidthSelectionCriterionType bandwidthSelectionCriterion() const;
    void setBandwidthSelectionCriterion(BandwidthSelectionCriterionType type);

public:     // Implement CGwmAlgorithm
    void run() override;

public:     // Implement IGwmRegressionAnalysis
    mat regression(const mat& x, const vec& y) override;
    mat regressionHatmatrix(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qdiag, mat& S) override;

    mat regressionSerial(const mat& x, const vec& y);
    mat regressionHatmatrixSerial(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qDiag, mat& S);

public:     // Implement IGwmBandwidthSelectable
    double getCriterion(CGwmBandwidthWeight* weight);

    double bandwidthSizeCriterionCVSerial(CGwmBandwidthWeight* bandwidthWeight);
    double bandwidthSizeCriterionAICSerial(CGwmBandwidthWeight* bandwidthWeight);

protected:
    bool isStoreS();

    void createRegressionDistanceParameter();
    void createPredictionDistanceParameter();

    void createResultLayer(initializer_list<ResultLayerDataItem> items);

protected:
    bool mHasHatMatrix = true;
    bool mHasFTest = false;
    bool mHasPredict = false;

    DistanceParameter* mRegressionDistanceParameter = nullptr;
    DistanceParameter* mPredictionDistanceParameter = nullptr;

    bool mIsAutoselectBandwidth = false;
    BandwidthSelectionCriterionType mBandwidthSelectionCriterion = BandwidthSelectionCriterionType::AIC;
    BandwidthSelectionCriterionCalculator mBandwidthSelectionCriterionFunction = &CGwmGWRBasic::bandwidthSizeCriterionCVSerial;
    BandwidthCriterionList mBandwidthSelectionCriterionList;

    RegressionCalculator mPredictFunction = &CGwmGWRBasic::regressionSerial;
    RegressionHatmatrixCalculator mRegressionHatmatrixFunction = &CGwmGWRBasic::regressionHatmatrixSerial;
};

inline mat CGwmGWRBasic::regression(const mat& x, const vec& y)
{
    return regressionSerial(x, y);
}

inline mat CGwmGWRBasic::regressionHatmatrix(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qdiag, mat& S)
{
    return regressionHatmatrixSerial(x, y, betasSE, shat, qdiag, S);
}

inline bool CGwmGWRBasic::isStoreS()
{
    return mHasHatMatrix && (mSourceLayer->featureCount() < 8192);
}

inline bool CGwmGWRBasic::isAutoselectBandwidth() const
{
    return mIsAutoselectBandwidth;
}

inline void CGwmGWRBasic::setIsAutoselectBandwidth(bool isAutoSelect)
{
    mIsAutoselectBandwidth = isAutoSelect;
}

inline CGwmGWRBasic::BandwidthSelectionCriterionType CGwmGWRBasic::bandwidthSelectionCriterion() const
{
    return mBandwidthSelectionCriterion;
}

inline double CGwmGWRBasic::getCriterion(CGwmBandwidthWeight* weight)
{
    return (this->*mBandwidthSelectionCriterionFunction)(weight);
}

#endif  // CGWMGWRBASIC_H