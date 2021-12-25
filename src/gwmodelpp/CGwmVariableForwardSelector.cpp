#include "CGwmVariableForwardSelector.h"

#include <armadillo>

using namespace arma;

CGwmVariableForwardSelector::CGwmVariableForwardSelector()
{
}

CGwmVariableForwardSelector::CGwmVariableForwardSelector(const vector<GwmVariable>& variables, double threshold)
    : mVariables(variables)
    , mThreshold(threshold)
{
}

CGwmVariableForwardSelector::~CGwmVariableForwardSelector()
{
}

vector<GwmVariable> CGwmVariableForwardSelector::optimize(IGwmVarialbeSelectable *instance)
{
    vector<int> curIndex, restIndex;
    for (size_t i = 0; i < mVariables.size(); i++)
    {
        restIndex.push_back(i);
    }
    vector<pair<vector<int>, double> > modelCriterions;
    for (size_t i = 0; i < mVariables.size(); i++)
    {
        vec criterions = vec(mVariables.size() - i);
        for (size_t j = 0; j < restIndex.size(); j++)
        {
            curIndex.push_back(restIndex[j]);
            double aic = instance->getCriterion(convertIndexToVariables(curIndex));
            criterions(j) = aic;
            modelCriterions.push_back(make_pair(curIndex, aic));
            curIndex.pop_back();
        }
        uword iBestVar = criterions.index_min();
        curIndex.push_back(restIndex[iBestVar]);
        restIndex.erase(restIndex.begin() + iBestVar);
    }
    mVarsCriterion = sort(modelCriterions);
    return convertIndexToVariables(select(mVarsCriterion).first);
}

vector<GwmVariable> CGwmVariableForwardSelector::convertIndexToVariables(vector<int> index)
{
    vector<GwmVariable> variables;
    for (int i : index)
        variables.push_back(mVariables[i]);
    return variables;
}

vector<pair<vector<int>, double> > CGwmVariableForwardSelector::sort(vector<pair<vector<int>, double> > models)
{
    size_t tag = 0;
    vector<int> sortIndex;
    for (size_t i = mVariables.size(); i > 0; i--)
    {
        std::sort(models.begin() + tag, models.begin() + tag + i, [](const pair<vector<int>, double>& left, const pair<vector<int>, double>& right){
            return left.second > right.second;
        });
        tag += i;
    }
    return models;
}

pair<vector<int>, double> CGwmVariableForwardSelector::select(vector<pair<vector<int>, double> > models)
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

vector<pair<vector<GwmVariable>, double> > CGwmVariableForwardSelector::indepVarsCriterion() const
{
    vector<pair<vector<GwmVariable>, double> > criterions;
    for (pair<vector<int>, double> item : mVarsCriterion)
    {
        vector<GwmVariable> variables;
        for (int i : item.first)
        {
            variables.push_back(mVariables[i]);
        }
        criterions.push_back(make_pair(variables, item.second));
    }
    return criterions;
}
