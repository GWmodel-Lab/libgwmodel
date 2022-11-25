#ifndef CGWMGTWR_H
#define CGWMGTWR_H

#include <utility>
#include <string>
#include <initializer_list>
#include "CGwmGWRBase.h"
#include "GwmRegressionDiagnostic.h"
#include "IGwmBandwidthSelectable.h"
#include "IGwmVarialbeSelectable.h"
#include "IGwmParallelizable.h"

#include "spatialweight/CGwmCRSTDistance.h"

using namespace std;

class CGwmGTWR :  public CGwmGWRBase, public IGwmBandwidthSelectable, public IGwmVarialbeSelectable, public IGwmOpenmpParallelizable
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
    
    typedef mat (CGwmGTWR::*RegressionCalculator)(const mat&, const vec&);
    typedef mat (CGwmGTWR::*RegressionHatmatrixCalculator)(const mat&, const vec&, mat&, vec&, vec&, mat&);

    typedef tuple<string, mat, NameFormat> ResultLayerDataItem;

    typedef double (CGwmGTWR::*BandwidthSelectionCriterionCalculator)(CGwmBandwidthWeight*);
    typedef double (CGwmGTWR::*IndepVarsSelectCriterionCalculator)(const vector<GwmVariable>&);

private:
    static GwmRegressionDiagnostic CalcDiagnostic(const mat& x, const vec& y, const mat& betas, const vec& shat);

public:
    CGwmGTWR();
    ~CGwmGTWR();

private:
    CGwmWeight* mWeight = nullptr;      
    CGwmDistance* mDistance = nullptr;  

//public:
//    vec weightVector(uword focus);//recalculate weight using spatial temporal distance

public:
    bool isAutoselectBandwidth() const;
    void setIsAutoselectBandwidth(bool isAutoSelect);

    BandwidthSelectionCriterionType bandwidthSelectionCriterion() const;
    void setBandwidthSelectionCriterion(const BandwidthSelectionCriterionType& criterion);

    bool isAutoselectIndepVars() const;
    void setIsAutoselectIndepVars(bool isAutoSelect);

    double indepVarSelectionThreshold() const;
    void setIndepVarSelectionThreshold(double threshold);
    
    VariablesCriterionList indepVarsSelectionCriterionList() const;
    BandwidthCriterionList bandwidthSelectionCriterionList() const;

    bool hasHatMatrix() const;
    void setHasHatMatrix(const bool has);

public:     // Implement CGwmAlgorithm
    void run() override;

public:     // Implement IGwmRegressionAnalysis
    mat regression(const mat& x, const vec& y) override;
    mat regressionHatmatrix(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qdiag, mat& S) override;
private:
    mat regressionSerial(const mat& x, const vec& y);
    mat regressionHatmatrixSerial(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qDiag, mat& S);
    
#ifdef ENABLE_OPENMP
    mat regressionOmp(const mat& x, const vec& y);
    mat regressionHatmatrixOmp(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qDiag, mat& S);
#endif

public:     // Implement IGwmBandwidthSelectable
    double getCriterion(CGwmBandwidthWeight* weight);
private:
    double bandwidthSizeCriterionCVSerial(CGwmBandwidthWeight* bandwidthWeight);
    double bandwidthSizeCriterionAICSerial(CGwmBandwidthWeight* bandwidthWeight);
#ifdef ENABLE_OPENMP
    double bandwidthSizeCriterionCVOmp(CGwmBandwidthWeight* bandwidthWeight);
    double bandwidthSizeCriterionAICOmp(CGwmBandwidthWeight* bandwidthWeight);
#endif

public:     // Implement IGwmVariableSelectable
    double getCriterion(const vector<GwmVariable>& variables);
private:
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

    void createDistanceParameter();

    void createPredictionDistanceParameter();

    void createResultLayer(initializer_list<ResultLayerDataItem> items);

protected:
    bool mHasHatMatrix = true;
    bool mHasFTest = false;
    bool mHasPredict = false;
    
    bool mIsAutoselectIndepVars = false;
    double mIndepVarSelectionThreshold = 3.0;
    IndepVarsSelectCriterionCalculator mIndepVarsSelectionCriterionFunction = &CGwmGTWR::indepVarsSelectionCriterionSerial;
    VariablesCriterionList mIndepVarsSelectionCriterionList;

    bool mIsAutoselectBandwidth = false;
    BandwidthSelectionCriterionType mBandwidthSelectionCriterion = BandwidthSelectionCriterionType::AIC;
    BandwidthSelectionCriterionCalculator mBandwidthSelectionCriterionFunction = &CGwmGTWR::bandwidthSizeCriterionCVSerial;
    BandwidthCriterionList mBandwidthSelectionCriterionList;

    RegressionCalculator mPredictFunction = &CGwmGTWR::regressionSerial;
    RegressionHatmatrixCalculator mRegressionHatmatrixFunction = &CGwmGTWR::regressionHatmatrixSerial;

    ParallelType mParallelType = ParallelType::SerialOnly;
    int mOmpThreadNum = 8;
};

inline mat CGwmGTWR::regression(const mat& x, const vec& y)
{
    return regressionSerial(x, y);
}

inline mat CGwmGTWR::regressionHatmatrix(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qdiag, mat& S)
{
    return regressionHatmatrixSerial(x, y, betasSE, shat, qdiag, S);
}

inline bool CGwmGTWR::isStoreS()
{
    return mHasHatMatrix && (mSourceLayer->featureCount() < 8192);
}

inline bool CGwmGTWR::isAutoselectBandwidth() const
{
    return mIsAutoselectBandwidth;
}

inline void CGwmGTWR::setIsAutoselectBandwidth(bool isAutoSelect)
{
    mIsAutoselectBandwidth = isAutoSelect;
}

inline bool CGwmGTWR::isAutoselectIndepVars() const
{
    return mIsAutoselectIndepVars;
}

inline void CGwmGTWR::setIsAutoselectIndepVars(bool isAutoSelect)
{
    mIsAutoselectIndepVars = isAutoSelect;
}

inline double CGwmGTWR::indepVarSelectionThreshold() const
{
    return mIndepVarSelectionThreshold;
}

inline void CGwmGTWR::setIndepVarSelectionThreshold(double threshold)
{
    mIndepVarSelectionThreshold = threshold;
}

inline CGwmGTWR::BandwidthSelectionCriterionType CGwmGTWR::bandwidthSelectionCriterion() const
{
    return mBandwidthSelectionCriterion;
}
    
inline VariablesCriterionList CGwmGTWR::indepVarsSelectionCriterionList() const
{
    return mIndepVarsSelectionCriterionList;
}

inline BandwidthCriterionList CGwmGTWR::bandwidthSelectionCriterionList() const
{
    return mBandwidthSelectionCriterionList;
}

inline double CGwmGTWR::getCriterion(CGwmBandwidthWeight* weight)
{
    return (this->*mBandwidthSelectionCriterionFunction)(weight);
}

inline double CGwmGTWR::getCriterion(const vector<GwmVariable>& variables)
{
    return (this->*mIndepVarsSelectionCriterionFunction)(variables);
}

inline int CGwmGTWR::parallelAbility() const
{
    return ParallelType::SerialOnly
#ifdef ENABLE_OPENMP
        | ParallelType::OpenMP
#endif        
        ;
}

inline ParallelType CGwmGTWR::parallelType() const
{
    return mParallelType;
}

inline void CGwmGTWR::setOmpThreadNum(const int threadNum)
{
    mOmpThreadNum = threadNum;
}

inline bool CGwmGTWR::hasHatMatrix() const
{
    return mHasHatMatrix;
}

inline void CGwmGTWR::setHasHatMatrix(const bool has)
{
    mHasHatMatrix = has;
}

inline void CGwmGTWR::createDistanceParameter()
{
    if (mSpatialWeight.distance()->type() == CGwmDistance::DistanceType::CRSDistance ||
        mSpatialWeight.distance()->type() == CGwmDistance::DistanceType::CRSTDistance || 
        mSpatialWeight.distance()->type() == CGwmDistance::DistanceType::MinkwoskiDistance)
    {
        mSpatialWeight.distance()->makeParameter({
            mSourceLayer->points(),
            mSourceLayer->points()
        });
    }
}

#endif  // CGWMGTWR_H