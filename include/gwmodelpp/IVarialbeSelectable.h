#ifndef IVARIALBESELECTABLE_H
#define IVARIALBESELECTABLE_H

#include <vector>

namespace gwm
{

struct IVarialbeSelectable
{
    
    /**
     * \~english
     * @brief Get criterion value with given variables for variable optimization.
     * 
     * @param weight Given variables
     * @return double Criterion value
     * 
     * \~chinese
     * @brief 根据指定的变量计算变量优选的指标值。
     * 
     * @param weight 指定的变量。
     * @return double 变量优选的指标值。
     */
    virtual double getCriterion(const std::vector<std::size_t>& variables) = 0;

    /**
     * \~english
     * @brief Get selected variables.
     * 
     * @return std::vector<std::size_t> Selected variables.
     * 
     * \~chinese
     * @brief 获取优选的变量。
     * 
     * @return std::vector<std::size_t> 优选的变量。
     */
    virtual std::vector<std::size_t> selectedVariables() = 0;
    
};

typedef std::vector<std::pair<std::vector<std::size_t>, double> > VariablesCriterionList;

}

#endif  // IVARIALBESELECTABLE_H