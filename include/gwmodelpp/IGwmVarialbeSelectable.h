#ifndef IGWMVARIALBESELECTABLE_H
#define IGWMVARIALBESELECTABLE_H

#include <vector>

namespace gwm
{

struct IGwmVarialbeSelectable
{
    virtual double getCriterion(const std::vector<std::size_t>& variables) = 0;

    /**
     * @brief Get result of vaiable selection.
     * 
     * @return std::vector<std::size_t> Selected variables.
     */
    virtual std::vector<std::size_t> selectedVariables() = 0;
};

typedef std::vector<std::pair<std::vector<std::size_t>, double> > VariablesCriterionList;

}

#endif  // IGWMVARIALBESELECTABLE_H