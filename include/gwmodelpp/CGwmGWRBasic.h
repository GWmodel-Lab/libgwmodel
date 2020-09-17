#ifndef CGWMGWRBASIC_H
#define CGWMGWRBASIC_H

#include <utility>
#include <string>
#include <initializer_list>
#include "gwmodelpp.h"
#include "CGwmGWRBase.h"
#include "GwmRegressionDiagnostic.h"
#include "IGwmBandwidthSelectable.h"
#include "IGwmVarialbeSelectable.h"
#include "IGwmParallelizable.h"

using namespace std;

class GWMODELPP_API CGwmGWRBasic : public CGwmGWRBase, public IGwmBandwidthSelectable, public IGwmVarialbeSelectable, public IGwmOpenmpParallelizable
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
    typedef double (CGwmGWRBasic::*IndepVarsSelectCriterionCalculator)(const vector<GwmVariable>&);

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
    
#ifdef ENABLE_OPENMP
    mat regressionOmp(const mat& x, const vec& y);
    mat regressionHatmatrixOmp(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qDiag, mat& S);
#endif

public:     // Implement IGwmBandwidthSelectable
    double getCriterion(CGwmBandwidthWeight* weight);

    double bandwidthSizeCriterionCVSerial(CGwmBandwidthWeight* bandwidthWeight);
    double bandwidthSizeCriterionAICSerial(CGwmBandwidthWeight* bandwidthWeight);
#ifdef ENABLE_OPENMP
    double bandwidthSizeCriterionCVOmp(CGwmBandwidthWeight* bandwidthWeight);
    double bandwidthSizeCriterionAICOmp(CGwmBandwidthWeight* bandwidthWeight);
#endif

public:     // Implement IGwmVariableSelectable
    double getCriterion(const vector<GwmVariable>& variables);

    double indepVarsSelectionCriterionSerial(const vector<GwmVariable>& indepVars);
#ifdef ENABLE_OPENMP
    double indepVarsSelectionCriterionOmp(const vector<GwmVariable>& indepVars);
#endif    

public:     // Implement IGwmParallelizable
    int parallelAbility() const;
    ParallelType parallelType() const;
    void setParallelType(const ParallelType& type);

public:     // Implement IGwmOpenmpParallelizable
    void setOmpThreadNum(const int threadNum);

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

    bool mIsAutoselectIndepVars = false;
    double mIndepVarSelectionThreshold = 3.0;
    IndepVarsSelectCriterionCalculator mIndepVarsSelectionCriterionFunction = &CGwmGWRBasic::indepVarsSelectionCriterionSerial;
    VariablesCriterionList mIndepVarsSelectionCriterionList;

    bool mIsAutoselectBandwidth = false;
    BandwidthSelectionCriterionType mBandwidthSelectionCriterion = BandwidthSelectionCriterionType::AIC;
    BandwidthSelectionCriterionCalculator mBandwidthSelectionCriterionFunction = &CGwmGWRBasic::bandwidthSizeCriterionCVSerial;
    BandwidthCriterionList mBandwidthSelectionCriterionList;

    RegressionCalculator mPredictFunction = &CGwmGWRBasic::regressionSerial;
    RegressionHatmatrixCalculator mRegressionHatmatrixFunction = &CGwmGWRBasic::regressionHatmatrixSerial;

    ParallelType mParallelType = ParallelType::SerialOnly;
    int mOmpThreadNum = 8;
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

inline double CGwmGWRBasic::getCriterion(const vector<GwmVariable>& variables)
{
    return (this->*mIndepVarsSelectionCriterionFunction)(variables);
}

inline int CGwmGWRBasic::parallelAbility() const
{
    return ParallelType::SerialOnly
#ifdef ENABLE_OPENMP
        | ParallelType::OpenMP
#endif        
        ;
}

inline ParallelType CGwmGWRBasic::parallelType() const
{
    return mParallelType;
}

inline void CGwmGWRBasic::setOmpThreadNum(const int threadNum)
{
    mOmpThreadNum = threadNum;
}

#endif  // CGWMGWRBASIC_H