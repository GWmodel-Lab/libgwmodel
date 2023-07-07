#ifndef IVARIALBESELECTABLE_H
#define IVARIALBESELECTABLE_H

#include <vector>
#include <sstream>
#include "Status.h"
#include "Logger.h"

namespace gwm
{

#define GWM_LOG_TAG_VARIABLE_CRITERION "#variable-criterion "

/**
 * @brief \~english Interface for variable selectable algorithms. \~chinese 可变量优选接口。
 * 
 */
struct IVarialbeSelectable
{
    
    /**
     * @brief \~english Get meta infomation of variable combination and the corresponding criterion value.
     * \~chinese 获取当前带宽值和对应指标值的元信息。
     * 
     * @return std::stringstream \~english Stream of information string \~chinese 信息字符串流
     */
    static std::stringstream infoVariableCriterion()
    {
        return std::stringstream() << GWM_LOG_TAG_VARIABLE_CRITERION << "variables" << "," << "criterion";
    }

    /**
     * @brief \~english Get infomation of variable combination and the corresponding criterion value.
     * \~chinese 获取当前带宽值和对应指标值的信息。
     * 
     * @param variables \~english Variable combination \~chinese 变量组合
     * @param criterion \~english Criterion value \~chinese 指标值
     * @return std::stringstream \~english Stream of information string \~chinese 信息字符串流
     */
    static std::stringstream infoVariableCriterion(const std::vector<std::size_t>& variables, const double criterion)
    {
        std::vector<std::string> var_labels(variables.size());
        std::transform(variables.cbegin(), variables.cend(), var_labels.begin(), [](const std::size_t& var)
        {
            return std::to_string(var);
        });
        return std::stringstream() << "#variable-criterion " << strjoin("+", var_labels) << "," << criterion;
    }
    
    /**
     * \~english
     * @brief Get criterion value with given variables for variable optimization.
     * 
     * @param variables Given variables
     * @param criterion [out] Criterion value.
     * @return Status Algorithm status.
     * 
     * \~chinese
     * @brief 根据指定的变量计算变量优选的指标值。
     * 
     * @param variables 指定的变量。
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