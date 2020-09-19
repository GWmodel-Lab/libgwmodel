#ifndef IGWMVARIALBESELECTABLE_H
#define IGWMVARIALBESELECTABLE_H

#include <vector>
#include "GwmVariable.h"

using namespace std;

struct IGwmVarialbeSelectable
{
    virtual double getCriterion(const vector<GwmVariable>& variables) = 0;
};

typedef vector<pair<vector<GwmVariable>, double> > VariablesCriterionList;


#endif  // IGWMVARIALBESELECTABLE_H