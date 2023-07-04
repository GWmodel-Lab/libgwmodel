#include "VariableForwardSelector.h"

#include <armadillo>

using namespace arma;
using namespace std;
using namespace gwm;

vector<size_t> VariableForwardSelector::optimize(IVarialbeSelectable *instance)
{
    vector<size_t> curIndex, restIndex;
    for (size_t i = 0; i < mVariables.size(); i++)
    {
        restIndex.push_back(i);
    }
    vector<pair<vector<size_t>, double> > modelCriterions;
    Status status = Status::Success;
    for (size_t i = 0; i < mVariables.size(); i++)
    {
        if (status != Status::Success) break;
        vec criterions = vec(mVariables.size() - i);
        for (size_t j = 0; j < restIndex.size(); j++)
        {
            if (status != Status::Success) break;
            curIndex.push_back(restIndex[j]);
            double aic = DBL_MAX;
            status = instance->getCriterion(convertIndexToVariables(curIndex), aic);
            criterions(j) = aic;
            modelCriterions.push_back(make_pair(curIndex, aic));
            curIndex.pop_back();
        }
        uword iBestVar = criterions.index_min();
        curIndex.push_back(restIndex[iBestVar]);
        restIndex.erase(restIndex.begin() + iBestVar);
    }
    if (status == Status::Success)
    {
        mVarsCriterion = sort(modelCriterions);
        return convertIndexToVariables(select(mVarsCriterion).first);
    }
    else return mVariables;
}

std::vector<std::size_t> VariableForwardSelector::convertIndexToVariables(std::vector<std::size_t> index)
{
    vector<size_t> variables;
    for (size_t i : index)
        variables.push_back(mVariables[i]);
    return variables;
}

vector<pair<vector<size_t>, double> > VariableForwardSelector::sort(vector<pair<vector<size_t>, double> > models)
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

pair<vector<size_t>, double> VariableForwardSelector::select(vector<pair<vector<size_t>, double> > models)
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

VariablesCriterionList VariableForwardSelector::indepVarsCriterion() const
{
    VariablesCriterionList criterions;
    for (pair<vector<size_t>, double> item : mVarsCriterion)
    {
        vector<size_t> variables;
        for (size_t i : item.first)
        {
            variables.push_back(mVariables[i]);
        }
        criterions.push_back(make_pair(variables, item.second));
    }
    return criterions;
}
