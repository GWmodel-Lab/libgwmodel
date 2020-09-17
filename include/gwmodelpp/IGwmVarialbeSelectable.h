#ifndef IGWMVARIALBESELECTABLE_H
#define IGWMVARIALBESELECTABLE_H

#include <vector>
#include "gwmodelpp.h"
#include "GwmVariable.h"

using namespace std;

struct GWMODELPP_API IGwmVarialbeSelectable
{
    virtual double getCriterion(const vector<GwmVariable>& variables) = 0;
};

typedef vector<pair<vector<GwmVariable>, double> > VariablesCriterionList;


#endif  // IGWMVARIALBESELECTABLE_H