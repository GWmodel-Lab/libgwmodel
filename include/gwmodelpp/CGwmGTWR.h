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

class CGwmGTWR :  public CGwmGWRBase, public IGwmBandwidthSelectable, public IGwmVarialbeSelectable
{
public:
    enum BandwidthSelectionCriterionType
    {
        AIC,
        CV
    };

    static std::unordered_map<BandwidthSelectionCriterionType, std::string> BandwidthSelectionCriterionTypeNameMapper;
    
    typedef arma::mat (CGwmGTWR::*PredictCalculator)(const arma::mat&, const arma::mat&, const arma::vec&);
    typedef arma::mat (CGwmGTWR::*FitCalculator)(const arma::mat&, const arma::vec&, arma::mat&, arma::vec&, arma::vec&, arma::mat&);

    typedef double (CGwmGTWR::*BandwidthSelectionCriterionCalculator)(CGwmBandwidthWeight*);
    typedef double (CGwmGTWR::*IndepVarsSelectCriterionCalculator)(const std::vector<std::size_t>&);

private:
    static GwmRegressionDiagnostic CalcDiagnostic(const arma::mat& x, const arma::vec& y, const arma::mat& betas, const arma::vec& shat);

public:
    CGwmGTWR(){};
    CGwmGTWR(const arma::mat& x, const arma::vec& y, const arma::mat& coords, const CGwmSpatialWeight& spatialWeight, bool hasHatMatrix = true, bool hasIntercept = true)
        : CGwmGWRBase(x, y, spatialWeight, coords)
    {
        mHasHatMatrix = hasHatMatrix;
        mHasIntercept = hasIntercept;
    }
    ~CGwmGTWR(){};

private:
    CGwmWeight* mWeight = nullptr;      
    CGwmDistance* mDistance = nullptr;  

//public:
//    arma::vec weightVector(uword focus);//recalculate weight using spatial temporal distance

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
    arma::mat predict(const arma::mat& locations) override;

    arma::mat fit() override;

private:
    arma::mat predictSerial(const arma::mat& locations, const arma::mat& x, const arma::vec& y);
    arma::mat fitSerial(const arma::mat& x, const arma::vec& y, arma::mat& betasSE, arma::vec& shat, arma::vec& qDiag, arma::mat& S);
    

public:     // Implement IGwmBandwidthSelectable
    double getCriterion(CGwmBandwidthWeight* weight) override
    {
        return (this->*mBandwidthSelectionCriterionFunction)(weight);
    }

private:
    double bandwidthSizeCriterionCVSerial(CGwmBandwidthWeight* bandwidthWeight);
    double bandwidthSizeCriterionAICSerial(CGwmBandwidthWeight* bandwidthWeight);


public:     // Implement IGwmVariableSelectable
    double getCriterion(const std::vector<std::size_t>& variables) override
    {
        return (this->*mIndepVarsSelectionCriterionFunction)(variables);
    }

    std::vector<std::size_t> selectedVariables() override
    {
        return mSelectedIndepVars;
    }

private:
    double indepVarsSelectionCriterionSerial(const std::vector<std::size_t>& indepVars);

protected:
    bool isStoreS() { return mHasHatMatrix && (mCoords.n_rows < 8192); }

    void createPredictionDistanceParameter(const arma::mat& locations);
    void createDistanceParameter();

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

    // ParallelType mParallelType = ParallelType::SerialOnly;

    arma::mat mBetasSE;
    arma::vec mSHat;
    arma::vec mQDiag;
    arma::mat mS;

};

#endif  // CGWMGTWR_H