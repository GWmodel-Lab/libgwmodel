#ifndef CGWMGWRBASIC_H
#define CGWMGWRBASIC_H

#include "CGwmGWRBase.h"

class CGwmGWRBasic : public CGwmGWRBase
{
public:
    enum BandwidthSelectionCriterionType
    {
        AIC,
        CV
    };

    static unordered_map<BandwidthSelectionCriterionType, string> BandwidthSelectionCriterionTypeNameMapper;
    
    typedef mat (CGwmGWRBasic::*RegressionCalculator)(const mat&, const vec&, mat&, vec&, vec&, mat&);

public:
    CGwmGWRBasic();
    ~CGwmGWRBasic();

public:
    mat regression(const mat& x, const vec& y) override;

protected:
    mat regressionSerial(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qDiag, mat& S);

protected:
    bool mHasHatMatrix = true;
    bool mHasFTest = false;
    bool mHasPredict = false;

    vec mQDiag;
    mat mBetasSE;

    vec mShat;
    mat mS;

    RegressionCalculator mRegressionFunction = &CGwmGWRBasic::regressionSerial;
    // RegressionHatmatrix mPredictFunction = &CGwmGWRBasic::regressionHatmatrixSerial;
};

#endif  // CGWMGWRBASIC_H