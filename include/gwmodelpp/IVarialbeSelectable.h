#ifndef IVARIALBESELECTABLE_H
#define IVARIALBESELECTABLE_H

#include <vector>
#include "Status.h"

namespace gwm
{

#define GWM_LOG_TAG_VARIABLE_CRITERION "#variable-criterion "

/**
 * @brief \~english Interface for variable selectable algorithms. \~chinese 可变量优选接口。
 * 
 */
struct IVarialbeSelectable
{
    static std::stringstream infoVariableCriterion()
    {
        return std::stringstream() << GWM_LOG_TAG_VARIABLE_CRITERION << "variables" << "," << "criterion";
    }

    static std::stringstream infoVariableCriterion(const std::vector<std::size_t>& variables, const double criterion)
    {
        std::string formula = std::accumulate(variables.cbegin(), variables.cend(), std::string(""), [](std::string x0, std::size_t x)
        {
            return x0 + "+" + std::to_string(x);
        });
        return std::stringstream() << "#variable-criterion " << formula << "," << criterion;
    }
    
    /**
     * \~english
     * @brief Get criterion value with given variables for variable optimization.
     * 
     * @param weight Given variables
     * @param criterion [out] Criterion value.
     * @return Status Algorithm status.
     * 
     * \~chinese
     * @brief 根据指定的变量计算变量优选的指标值。
     * 
     * @param weight 指定的变量。
     * @param criterion [出参] 带宽优选的指标值。
     * @param Status 算法运行状态。
     */
    virtual Status getCriterion(const std::vector<std::size_t>& variables, double& criterion) = 0;

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