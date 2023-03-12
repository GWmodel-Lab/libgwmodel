#ifndef GTWR_H
#define GTWR_H

#include <utility>
#include <string>
#include <initializer_list>
#include "GWRBase.h"
#include "RegressionDiagnostic.h"
#include "IBandwidthSelectable.h"
#include "IVarialbeSelectable.h"
#include "IParallelizable.h"
#include "spatialweight/CRSSTDistance.h"

namespace gwm
{

class GTWR :  public GWRBase, public IBandwidthSelectable, public IVarialbeSelectable
{
public:
    enum BandwidthSelectionCriterionType
    {
        AIC,
        CV
    };

    static std::unordered_map<BandwidthSelectionCriterionType, std::string> BandwidthSelectionCriterionTypeNameMapper;
    
    typedef arma::mat (GTWR::*PredictCalculator)(const arma::mat&, const arma::mat&, const arma::vec&);
    typedef arma::mat (GTWR::*FitCalculator)(const arma::mat&, const arma::vec&, arma::mat&, arma::vec&, arma::vec&, arma::mat&);

    typedef double (GTWR::*BandwidthSelectionCriterionCalculator)(BandwidthWeight*);
    typedef double (GTWR::*IndepVarsSelectCriterionCalculator)(const std::vector<std::size_t>&);

private:
    static RegressionDiagnostic CalcDiagnostic(const arma::mat& x, const arma::vec& y, const arma::mat& betas, const arma::vec& shat);

public:
    GTWR(){};
    GTWR(const arma::mat& x, const arma::vec& y, const arma::mat& coords, const SpatialWeight& spatialWeight, bool hasHatMatrix = true, bool hasIntercept = true)
        : GWRBase(x, y, spatialWeight, coords)
    {
        mHasHatMatrix = hasHatMatrix;
        mHasIntercept = hasIntercept;
    }
    ~GTWR(){};

private:
    Weight* mWeight = nullptr;      
    Distance* mDistance = nullptr;  

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

    // void setTimes(const arma::vec& times)
    // {
    //     vTimes=times;
    // }

    void setCoords(const arma::mat& coords, const arma::vec& times)
    {
        mCoords=coords;
        vTimes=times;
    }

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
    

public:     // Implement IBandwidthSelectable
    double getCriterion(BandwidthWeight* weight) override
    {
        return (this->*mBandwidthSelectionCriterionFunction)(weight);
    }

private:
    double bandwidthSizeCriterionCVSerial(BandwidthWeight* bandwidthWeight);
    double bandwidthSizeCriterionAICSerial(BandwidthWeight* bandwidthWeight);


public:     // Implement IVariableSelectable
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
    IndepVarsSelectCriterionCalculator mIndepVarsSelectionCriterionFunction = &GTWR::indepVarsSelectionCriterionSerial;
    VariablesCriterionList mIndepVarsSelectionCriterionList;
    std::vector<std::size_t> mSelectedIndepVars;

    bool mIsAutoselectBandwidth = false;
    BandwidthSelectionCriterionType mBandwidthSelectionCriterion = BandwidthSelectionCriterionType::AIC;
    BandwidthSelectionCriterionCalculator mBandwidthSelectionCriterionFunction = &GTWR::bandwidthSizeCriterionCVSerial;
    BandwidthCriterionList mBandwidthSelectionCriterionList;

    PredictCalculator mPredictFunction = &GTWR::predictSerial;
    FitCalculator mFitFunction = &GTWR::fitSerial;

    // ParallelType mParallelType = ParallelType::SerialOnly;

    arma::mat mBetasSE;
    arma::vec mSHat;
    arma::vec mQDiag;
    arma::mat mS;

    arma::vec vTimes;

};

}

#endif  // GTWR_H