#ifndef IGWMVARIALBESELECTABLE_H
#define IGWMVARIALBESELECTABLE_H

#include <vector>

struct IGwmVarialbeSelectable
{
    virtual double getCriterion(const std::vector<std::size_t>& variables) = 0;
};

typedef std::vector<std::pair<std::vector<std::size_t>, double> > VariablesCriterionList;


#endif  // IGWMVARIALBESELECTABLE_H