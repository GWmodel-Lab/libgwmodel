#ifndef VARIABLEFORWARDSELECTOR_H
#define VARIABLEFORWARDSELECTOR_H

#include <utility>
#include "armadillo_config.h"
#include "IVarialbeSelectable.h"

namespace gwm
{

/**
 * @brief \~english Forward variable selector. \~chinese 变量前向选择器。
 * 
 */
class VariableForwardSelector
{
public:

    /**
     * @brief \~english Convert index form type std::size_t to arma::uvec. \~chinese 将索引值的类型 std::size_t 转换为 arma::uvec 类型
     * 
     * @param index \~english Indeces of type std::size_t \~chinese 索引值（ std::size_t 类型）
     * @param hasIntercept \~english Whether has intercept \~chinese 是否有截距
     * @return arma::uvec \~english Indeces of type arma::uvec \~chinese 索引值（ arma::uvec 类型）
     */
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

    /**
     * @brief \~english Construct a new Variable Forward Selector object. \~chinese 构造一个新的 VariableForwardSelector 对象。
     */
    VariableForwardSelector() {}

    /**
     * @brief \~english Construct a new Variable Forward Selector object. \~chinese 构造一个新的 VariableForwardSelector 对象。
     * 
     * @param variables \~english Indeces \~chinese 索引值
     * @param threshold \~english Threshold \~chinese 阈值
     */
    VariableForwardSelector(const std::vector<std::size_t>& variables, double threshold) : mVariables(variables) , mThreshold(threshold) {}

    /**
     * @brief \~english Destroy the Variable Forward Selector object. \~chinese 销毁 VariableForwardSelector 对象。
     * 
     */
    ~VariableForwardSelector() {}

    /**
     * @brief \~english Get variables. \~chinese 获取变量。
     * 
     * @return std::vector<std::size_t> \~english Indeces \~chinese 索引值
     */
    std::vector<std::size_t> indepVars() const { return mVariables; }

    /**
     * @brief \~english Set variables. \~chinese 设置变量。
     * 
     * @param indepVars \~english Indeces \~chinese 索引值
     */
    void setIndepVars(const std::vector<std::size_t> &indepVars) { mVariables = indepVars; }

    /**
     * @brief \~english Get threshold. \~chinese 获取阈值。
     * 
     * @return double \~english Threshold \~chinese 阈值
     */
    double threshold() const { return mThreshold; }

    /**
     * @brief \~english Set threshold. \~chinese 设置阈值。
     * 
     * @param threshold \~english Threshold \~chinese 阈值
     */
    void setThreshold(double threshold) { mThreshold = threshold; }

public:

    /**
     * @brief \~english Optimize variable combination. \~chinese 优选变量组合。
     * 
     * @param instance \~english A pointer to a instance of type inherited from gwm::IVarialbeSelectable \~chinese 指向派生自 gwm::IVarialbeSelectable 类型对象的指针
     * @return std::vector<std::size_t> \~english Optimized variable combination \~chinese 优选后的变量组合
     */
    std::vector<std::size_t> optimize(IVarialbeSelectable* instance);

    /**
     * @brief \~english Get the list of criterion values for each variable combination in independent variable selection. \~chinese 获取变量优选过程中每种变量组合对应的指标值列表。
     * 
     * @return VariablesCriterionList \~english List of criterion values for each variable combination in independent variable selection \~chinese 变量优选过程中每种变量组合对应的指标值列表
     */
    VariablesCriterionList indepVarsCriterion() const;

private:

    /**
     * @deprecated
     */
    std::vector<std::size_t> convertIndexToVariables(std::vector<std::size_t> index);

    /**
     * @brief \~english Sort variable combinations. \~chinese 对变量组合进行排序。
     * 
     * @param models \~english The original list of variable combinations \~chinese 包含所有变量组合的原始列表
     * @return std::vector<std::pair<std::vector<std::size_t>, double> > \~english The sorted list of variable combinations \~chinese 排序后的包含所有变量组合的列表
     */
    std::vector<std::pair<std::vector<std::size_t>, double> > sort(std::vector<std::pair<std::vector<std::size_t>, double> > models);

    /**
     * @brief \~english Select the optimized variable combination. \~chinese 选择最优变量组合。
     * 
     * @param models \~english The original list of variable combinations \~chinese 包含所有变量组合的原始列表
     * @return std::pair<std::vector<std::size_t>, double> \~english The optimized variable combinations \~chinese 最优变量组合
     */
    std::pair<std::vector<std::size_t>, double> select(std::vector<std::pair<std::vector<std::size_t>, double> > models);

private:
    std::vector<std::size_t> mVariables;    //!< \~english Variables to be selected \~chinese 要优选的变量
    double mThreshold;                      //!< \~english Threshold \~chinese 阈值

    std::vector<std::pair<std::vector<std::size_t>, double> > mVarsCriterion;   //!< \~english List of criterion values for each variable combination in independent variable selection \~chinese 变量优选过程中每种变量组合对应的指标值列表
};

}

#endif  // VARIABLEFORWARDSELECTOR_H