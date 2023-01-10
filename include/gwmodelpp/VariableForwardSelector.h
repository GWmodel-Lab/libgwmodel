#ifndef VARIABLEFORWARDSELECTOR_H
#define VARIABLEFORWARDSELECTOR_H

#include <utility>
#include <armadillo>
#include "IVarialbeSelectable.h"

namespace gwm
{

class VariableForwardSelector
{
public:
    static arma::uvec index2uvec(const std::vector<std::size_t>& index, bool hasIntercept)
    {
        std::size_t start_index = hasIntercept ? 1 : 0;
        arma::uvec sel_indep_vars(index.size() + start_index, arma::fill::zeros);
        for (std::size_t i = 0; i < index.size(); i++)
        {
            sel_indep_vars(i + start_index) = index[i];
        }
        return sel_indep_vars;
    }

public:
    VariableForwardSelector() {}

    VariableForwardSelector(const std::vector<std::size_t>& variables, double threshold) : mVariables(variables) , mThreshold(threshold) {}

    ~VariableForwardSelector() {}

    std::vector<std::size_t> indepVars() const
    {
        return mVariables;
    }

    void setIndepVars(const std::vector<std::size_t> &indepVars)
    {
        mVariables = indepVars;
    }

    double threshold() const
    {
        return mThreshold;
    }

    void setThreshold(double threshold)
    {
        mThreshold = threshold;
    }

public:
    std::vector<std::size_t> optimize(IVarialbeSelectable* instance);
    VariablesCriterionList indepVarsCriterion() const;

private:
    std::vector<std::size_t> convertIndexToVariables(std::vector<std::size_t> index);
    std::vector<std::pair<std::vector<std::size_t>, double> > sort(std::vector<std::pair<std::vector<std::size_t>, double> > models);
    std::pair<std::vector<std::size_t>, double> select(std::vector<std::pair<std::vector<std::size_t>, double> > models);

private:
    std::vector<std::size_t> mVariables;
    double mThreshold;

    std::vector<std::pair<std::vector<std::size_t>, double> > mVarsCriterion;
};

}

#endif  // VARIABLEFORWARDSELECTOR_H