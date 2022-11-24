#include "CGwmVariableForwardSelector.h"

#include <armadillo>

using namespace arma;
using namespace std;

vector<size_t> CGwmVariableForwardSelector::optimize(IGwmVarialbeSelectable *instance)
{
    vector<size_t> curIndex, restIndex;
    for (int i = 0; i < mVariables.size(); i++)
    {
        restIndex.push_back(i);
    }
    vector<pair<vector<size_t>, double> > modelCriterions;
    for (int i = 0; i < mVariables.size(); i++)
    {
        vec criterions = vec(mVariables.size() - i);
        for (int j = 0; j < restIndex.size(); j++)
        {
            curIndex.push_back(restIndex[j]);
            double aic = instance->getCriterion(curIndex);
            criterions(j) = aic;
            modelCriterions.push_back(make_pair(curIndex, aic));
            curIndex.pop_back();
        }
        uword iBestVar = criterions.index_min();
        curIndex.push_back(restIndex[iBestVar]);
        restIndex.erase(restIndex.begin() + iBestVar);
    }
    mVarsCriterion = sort(modelCriterions);
    return select(mVarsCriterion).first;
}

vector<pair<vector<size_t>, double> > CGwmVariableForwardSelector::sort(vector<pair<vector<size_t>, double> > models)
{
    size_t tag = 0;
    vector<size_t> sortIndex;
    for (size_t i = mVariables.size(); i > 0; i--)
    {
        std::sort(models.begin() + tag, models.begin() + tag + i, [](const pair<vector<size_t>, double>& left, const pair<vector<size_t>, double>& right){
            return left.second > right.second;
        });
        tag += i;
    }
    return models;
}

pair<vector<size_t>, double> CGwmVariableForwardSelector::select(vector<pair<vector<size_t>, double> > models)
{
    for (size_t i = models.size() - 1; i >= 0; i--)
    {
        if (models[i - 1].second - models[i].second >= mThreshold)
        {
            return models[i];
        }
    }
    return models.back();
}

VariablesCriterionList CGwmVariableForwardSelector::indepVarsCriterion() const
{
    VariablesCriterionList criterions;
    for (pair<vector<size_t>, double> item : mVarsCriterion)
    {
        vector<size_t> variables;
        for (int i : item.first)
        {
            variables.push_back(mVariables[i]);
        }
        criterions.push_back(make_pair(variables, item.second));
    }
    return criterions;
}
