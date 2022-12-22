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

#include "spatialweight/CGwmCRSSTDistance.h"

using namespace std;

class CGwmGTWR :  public CGwmGWRBase, public IGwmBandwidthSelectable, public IGwmVarialbeSelectable, public IGwmOpenmpParallelizable
{
public:
    enum BandwidthSelectionCriterionType
    {
        AIC,
        CV
    };

    static unordered_map<BandwidthSelectionCriterionType, string> BandwidthSelectionCriterionTypeNameMapper;
    
    typedef mat (CGwmGTWR::*PredictCalculator)(const mat&, const mat&, const vec&);
    typedef mat (CGwmGTWR::*FitCalculator)(const mat&, const vec&, mat&, vec&, vec&, mat&);

    typedef double (CGwmGTWR::*BandwidthSelectionCriterionCalculator)(CGwmBandwidthWeight*);
    typedef double (CGwmGTWR::*IndepVarsSelectCriterionCalculator)(const std::vector<size_t>&);

private:
    static GwmRegressionDiagnostic CalcDiagnostic(const mat& x, const vec& y, const mat& betas, const vec& shat);

public:
    CGwmGTWR();
    CGwmGTWR(const mat& x, const vec& y, const mat& coords, const CGwmSpatialWeight& spatialWeight, bool hasHatMatrix = true, bool hasIntercept = true)
        : CGwmGWRBase(x, y, spatialWeight, coords)
    {
        mHasHatMatrix = hasHatMatrix;
        mHasIntercept = hasIntercept;
    }
    ~CGwmGTWR();

private:
    CGwmWeight* mWeight = nullptr;      
    CGwmDistance* mDistance = nullptr;  

//public:
//    vec weightVector(uword focus);//recalculate weight using spatial temporal distance

public:
    bool isAutoselectBandwidth() const { return mIsAutoselectBandwidth; }
    void setIsAutoselectBandwidth(bool isAutoSelect) { mIsAutoselectBandwidth = isAutoSelect; }

    BandwidthSelectionCriterionType bandwidthSelectionCriterion() const { return mBandwidthSelectionCriterion; }
    void setBandwidthSelectionCriterion(const BandwidthSelectionCriterionType& criterion);

    bool isAutoselectIndepVars() const { return mIsAutoselectIndepVars; }
    void setIsAutoselectIndepVars(bool isAutoSelect) { mIsAutoselectIndepVars = isAutoSelect; }

    double indepVarSelectionThreshold() const { return mIndepVarSelectionThreshold; }
    void setIndepVarSelectionThreshold(double threshold) { mIndepVarSelectionThreshold = threshold; }

    VariablesCriterionList indepVarsSelectionCriterionList() const { return mIndepVarsSelectionCriterionList; }
    BandwidthCriterionList bandwidthSelectionCriterionList() const { return mBandwidthSelectionCriterionList; }

    bool hasHatMatrix() const { return mHasHatMatrix; }
    void setHasHatMatrix(const bool has) { mHasHatMatrix = has; }

    arma::mat betasSE() { return mBetasSE; }
    arma::vec sHat() { return mSHat; }
    arma::vec qDiag() { return mQDiag; }
    arma::mat s() { return mS; }

public:     // Implement CGwmAlgorithm
    bool isValid() override;

public:     // Implement IGwmRegressionAnalysis
    mat predict(const mat& locations) override;

    mat fit() override;

private:
    mat predictSerial(const mat& locations, const mat& x, const vec& y);
    mat fitSerial(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qDiag, mat& S);
    
#ifdef ENABLE_OPENMP
    mat predictOmp(const mat& locations, const mat& x, const vec& y);
    mat fitOmp(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qDiag, mat& S);
#endif

public:     // Implement IGwmBandwidthSelectable
    double getCriterion(CGwmBandwidthWeight* weight) override
    {
        return (this->*mBandwidthSelectionCriterionFunction)(weight);
    }

private:
    double bandwidthSizeCriterionCVSerial(CGwmBandwidthWeight* bandwidthWeight);
    double bandwidthSizeCriterionAICSerial(CGwmBandwidthWeight* bandwidthWeight);
#ifdef ENABLE_OPENMP
    double bandwidthSizeCriterionCVOmp(CGwmBandwidthWeight* bandwidthWeight);
    double bandwidthSizeCriterionAICOmp(CGwmBandwidthWeight* bandwidthWeight);
#endif

public:     // Implement IGwmVariableSelectable
    double getCriterion(const std::vector<size_t>& variables) override
    {
        return (this->*mIndepVarsSelectionCriterionFunction)(variables);
    }

    std::vector<std::size_t> selectedVariables() override
    {
        return mSelectedIndepVars;
    }

private:
    double indepVarsSelectionCriterionSerial(const std::vector<size_t>& indepVars);
#ifdef ENABLE_OPENMP
    double indepVarsSelectionCriterionOmp(const std::vector<size_t>& indepVars);
#endif     

public:     // Implement IGwmParallelizable
    int parallelAbility() const
    {
        return ParallelType::SerialOnly
#ifdef ENABLE_OPENMP
            | ParallelType::OpenMP
#endif        
        ;
    }

    ParallelType parallelType() const { return mParallelType; }

    void setParallelType(const ParallelType& type);

public:     // Implement IGwmOpenmpParallelizable
    void setOmpThreadNum(const int threadNum) { mOmpThreadNum = threadNum; }

protected:
    bool isStoreS() { return mHasHatMatrix && (mCoords.n_rows < 8192); }

    void createPredictionDistanceParameter(const arma::mat& locations);

protected:
    bool mHasHatMatrix = true;
    bool mHasFTest = false;
    bool mHasPredict = false;
    
    bool mIsAutoselectIndepVars = false;
    double mIndepVarSelectionThreshold = 3.0;
    IndepVarsSelectCriterionCalculator mIndepVarsSelectionCriterionFunction = &CGwmGTWR::indepVarsSelectionCriterionSerial;
    VariablesCriterionList mIndepVarsSelectionCriterionList;
    std::vector<std::size_t> mSelectedIndepVars;

    bool mIsAutoselectBandwidth = false;
    BandwidthSelectionCriterionType mBandwidthSelectionCriterion = BandwidthSelectionCriterionType::AIC;
    BandwidthSelectionCriterionCalculator mBandwidthSelectionCriterionFunction = &CGwmGTWR::bandwidthSizeCriterionCVSerial;
    BandwidthCriterionList mBandwidthSelectionCriterionList;

    PredictCalculator mPredictFunction = &CGwmGTWR::predictSerial;
    FitCalculator mFitFunction = &CGwmGTWR::fitSerial;

    ParallelType mParallelType = ParallelType::SerialOnly;
    int mOmpThreadNum = 8;

    arma::mat mBetasSE;
    arma::vec mSHat;
    arma::vec mQDiag;
    arma::mat mS;

};

#endif  // CGWMGTWR_H