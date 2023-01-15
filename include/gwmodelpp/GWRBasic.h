#ifndef GWRBASIC_H
#define GWRBASIC_H

#include <utility>
#include <string>
#include <initializer_list>
#include "GWRBase.h"
#include "RegressionDiagnostic.h"
#include "IBandwidthSelectable.h"
#include "IVarialbeSelectable.h"
#include "IParallelizable.h"

namespace gwm
{

class GWRBasic : public GWRBase, public IBandwidthSelectable, public IVarialbeSelectable, public IOpenmpParallelizable
{
public:
    enum BandwidthSelectionCriterionType
    {
        AIC,
        CV
    };

    static std::unordered_map<BandwidthSelectionCriterionType, std::string> BandwidthSelectionCriterionTypeNameMapper;
    
    typedef arma::mat (GWRBasic::*PredictCalculator)(const arma::mat&, const arma::mat&, const arma::vec&);
    typedef arma::mat (GWRBasic::*FitCalculator)(const arma::mat&, const arma::vec&, arma::mat&, arma::vec&, arma::vec&, arma::mat&);

    typedef double (GWRBasic::*BandwidthSelectionCriterionCalculator)(BandwidthWeight*);
    typedef double (GWRBasic::*IndepVarsSelectCriterionCalculator)(const std::vector<size_t>&);

private:
    static RegressionDiagnostic CalcDiagnostic(const arma::mat& x, const arma::vec& y, const arma::mat& betas, const arma::vec& shat);

public:
    GWRBasic() {}
    GWRBasic(const arma::mat& x, const arma::vec& y, const arma::mat& coords, const SpatialWeight& spatialWeight, bool hasHatMatrix = true, bool hasIntercept = true)
        : GWRBase(x, y, spatialWeight, coords)
    {
        mHasHatMatrix = hasHatMatrix;
        mHasIntercept = hasIntercept;
    }
    ~GWRBasic() {}

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


public:     // Implement Algorithm
    bool isValid() override;

public:     // Implement IRegressionAnalysis
    arma::mat predict(const arma::mat& locations) override;

    arma::mat fit() override;

private:
    arma::mat predictSerial(const arma::mat& locations, const arma::mat& x, const arma::vec& y);
    arma::mat fitSerial(const arma::mat& x, const arma::vec& y, arma::mat& betasSE, arma::vec& shat, arma::vec& qDiag, arma::mat& S);
    
#ifdef ENABLE_OPENMP
    arma::mat predictOmp(const arma::mat& locations, const arma::mat& x, const arma::vec& y);
    arma::mat fitOmp(const arma::mat& x, const arma::vec& y, arma::mat& betasSE, arma::vec& shat, arma::vec& qDiag, arma::mat& S);
#endif

public:     // Implement IBandwidthSelectable
    double getCriterion(BandwidthWeight* weight) override
    {
        return (this->*mBandwidthSelectionCriterionFunction)(weight);
    }

private:
    double bandwidthSizeCriterionCVSerial(BandwidthWeight* bandwidthWeight);
    double bandwidthSizeCriterionAICSerial(BandwidthWeight* bandwidthWeight);
#ifdef ENABLE_OPENMP
    double bandwidthSizeCriterionCVOmp(BandwidthWeight* bandwidthWeight);
    double bandwidthSizeCriterionAICOmp(BandwidthWeight* bandwidthWeight);
#endif

public:     // Implement IVariableSelectable
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

public:     // Implement IParallelizable
    int parallelAbility() const override
    {
        return ParallelType::SerialOnly
#ifdef ENABLE_OPENMP
            | ParallelType::OpenMP
#endif        
        ;
    }

    ParallelType parallelType() const override { return mParallelType; }

    void setParallelType(const ParallelType& type) override;

public:     // Implement IOpenmpParallelizable
    void setOmpThreadNum(const int threadNum) override { mOmpThreadNum = threadNum; }

protected:
    bool isStoreS() { return mHasHatMatrix && (mCoords.n_rows < 8192); }

    void createPredictionDistanceParameter(const arma::mat& locations);

protected:
    bool mHasHatMatrix = true;
    bool mHasFTest = false;
    bool mHasPredict = false;
    
    bool mIsAutoselectIndepVars = false;
    double mIndepVarSelectionThreshold = 3.0;
    IndepVarsSelectCriterionCalculator mIndepVarsSelectionCriterionFunction = &GWRBasic::indepVarsSelectionCriterionSerial;
    VariablesCriterionList mIndepVarsSelectionCriterionList;
    std::vector<std::size_t> mSelectedIndepVars;

    bool mIsAutoselectBandwidth = false;
    BandwidthSelectionCriterionType mBandwidthSelectionCriterion = BandwidthSelectionCriterionType::AIC;
    BandwidthSelectionCriterionCalculator mBandwidthSelectionCriterionFunction = &GWRBasic::bandwidthSizeCriterionCVSerial;
    BandwidthCriterionList mBandwidthSelectionCriterionList;

    PredictCalculator mPredictFunction = &GWRBasic::predictSerial;
    FitCalculator mFitFunction = &GWRBasic::fitSerial;

    ParallelType mParallelType = ParallelType::SerialOnly;
    int mOmpThreadNum = 8;

    arma::mat mBetasSE;
    arma::vec mSHat;
    arma::vec mQDiag;
    arma::mat mS;
};

}

#endif  // GWRBASIC_H