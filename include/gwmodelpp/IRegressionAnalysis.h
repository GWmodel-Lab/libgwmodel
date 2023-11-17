#ifndef IREGRESSIONANALYSIS_H
#define IREGRESSIONANALYSIS_H

#include <vector>
#include <armadillo>
#include "RegressionDiagnostic.h"


namespace gwm
{

/**
 * @brief \~english Interface for regression analysis algorithms. \~chinese 回归分析算法接口。
 * 
 */
struct IRegressionAnalysis
{
    
    /**
     * \~english
     * @brief Get the Dependent Variable object.
     * 
     * @return arma::vec Dependent Variable.
     * 
     * \~chinese
     * @brief 获取因变量。
     * 
     * @return arma::vec 因变量。
     * 
     */
    virtual const arma::vec& dependentVariable() const = 0;

    /**
     * \~english
     * @brief Set the Dependent Variable object.
     * 
     * @param y 
     * 
     * \~chinese
     * @brief 设置因变量。
     * 
     * @param y 因变量。
     * 
     */
    virtual void setDependentVariable(const arma::vec& y) = 0;

    /**
     * \~english
     * @brief Get the Independent Variables object.
     * 
     * @return arma::mat Independent Variables.
     * 
     * \~chinese
     * @brief 获取自变量。
     * 
     * @return arma::mat 自变量。
     * 
     */
    virtual const arma::mat& independentVariables() const = 0;

    /**
     * \~english
     * @brief Set the Independent Variables object.
     * 
     * @param x 
     * 
     * \~chinese
     * @brief 设置自变量。
     * 
     * @param x 自变量。
     * 
     */
    virtual void setIndependentVariables(const arma::mat& x) = 0;
    
    /**
     * \~english
     * @brief Get whether has intercept.
     * 
     * @return true if has intercept.
     * @return false if doesnot has intercept.
     * 
     * \~chinese
     * @brief 获取是否具有截距。
     * 
     * @return true 如果有截距。
     * @return false 如果没有截距。
     * 
     */
    virtual bool hasIntercept() const = 0;

    /**
     * \~english
     * @brief Set the Has Intercept object.
     * 
     * @param has true if has intercept, otherwise false.
     * 
     * \~chinese
     * @brief Set the Has Intercept object.
     * 
     * @param has 如果有截距则传入 true ，否则传入 false 。
     * 
     */
    virtual void setHasIntercept(const bool has) = 0;

    /**
     * \~english
     * @brief Predict coefficients on specified locations.
     * 
     * @param locations Locations where to predict coefficients.
     * @return mat Predicted coefficients.
     * 
     * \~chinese
     * @brief 在指定位置处进行回归系数预测。
     * 
     * @param locations 指定位置。
     * @return mat 回归系数预测值。
     * 
     */
    virtual arma::mat predict(const arma::mat& locations) = 0;

    /**
     * \~english
     * @brief Fit coefficient estimates.
     * 
     * @return mat Coefficient estimates
     * 
     * \~chinese
     * @brief 拟合回归系数估计值。
     * 
     * @return mat 回归系数估计值。
     * 
     */
    virtual arma::mat fit() = 0;

    /**
     * \~english
     * @brief Get diagnostic information.
     * 
     * @return RegressionDiagnostic Diagnostic information.
     * 
     * \~chinese
     * @brief 获取诊断信息。
     * 
     * @return RegressionDiagnostic 诊断信息。
     */
    virtual RegressionDiagnostic diagnostic() const = 0;
};

}

#endif  // IREGRESSIONANALYSIS_H