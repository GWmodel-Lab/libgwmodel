#ifndef CGWMVARIABLEFORWARDSELECTOR_H
#define CGWMVARIABLEFORWARDSELECTOR_H

#include <utility>
#include "gwmodelpp.h"
#include "IGwmVarialbeSelectable.h"

using namespace std;

class GWMODELPP_API CGwmVariableForwardSelector
{
public:
    CGwmVariableForwardSelector();
    CGwmVariableForwardSelector(const vector<GwmVariable>& variables, double threshold);
    ~CGwmVariableForwardSelector();

    vector<GwmVariable> indepVars() const;
    void setIndepVars(const vector<GwmVariable> &indepVars);

    double threshold() const;
    void setThreshold(double threshold);

public:
    vector<GwmVariable> optimize(IGwmVarialbeSelectable* instance);
    VariablesCriterionList indepVarsCriterion() const;

private:
    vector<GwmVariable> convertIndexToVariables(vector<int> index);
    vector<pair<vector<int>, double> > sort(vector<pair<vector<int>, double> > models);
    pair<vector<int>, double> select(vector<pair<vector<int>, double> > models);

private:
    vector<GwmVariable> mVariables;
    double mThreshold;

    vector<pair<vector<int>, double> > mVarsCriterion;
};

inline vector<GwmVariable> CGwmVariableForwardSelector::indepVars() const
{
    return mVariables;
}

inline void CGwmVariableForwardSelector::setIndepVars(const vector<GwmVariable> &indepVars)
{
    mVariables = indepVars;
}

inline double CGwmVariableForwardSelector::threshold() const
{
    return mThreshold;
}

inline void CGwmVariableForwardSelector::setThreshold(double threshold)
{
    mThreshold = threshold;
}

#endif  // CGWMVARIABLEFORWARDSELECTOR_H