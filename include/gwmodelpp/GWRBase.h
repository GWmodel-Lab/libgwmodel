#ifndef GWRBASE_H
#define GWRBASE_H

#include "SpatialMonoscaleAlgorithm.h"
#include "IRegressionAnalysis.h"

namespace gwm
{

/**
 * \~english
 * @brief Interface for monoscale geographically weighted regression algorithm.
 * This class provides some commmon interfaces of geographically weighted regression algorithm.
 * It could not be constructed.
 * 
 * \~chinese
 * @brief 地理加权回归算法基类。
 * 该类提供一些地理加权回归算法的常用接口，不能被构造。
 * 
 */
class GWRBase : public SpatialMonoscaleAlgorithm, public IRegressionAnalysis
{
public:

    /**
     * \~english
     * @brief Calculate fitted values of dependent varialbe by given \f$X\f$ and \f$\beta\f$.
     * 
     * @param x Independent variables \f$X\f$.
     * @param betas Coefficient estimates \f$\beta\f$.
     * @return vec Fitted values of dependent varialbe.
     * 
     * \~chinese
     * @brief 根据给定的 \f$X\f$ 和 \f$\beta\f$ 计算拟合的因变量的值。
     * 
     * @param x 自变量矩阵 \f$X\f$。
     * @param betas 回归系数估计值 \f$\beta\f$。
     * @return vec 拟合的因变量的值。
     * 
     */
    static arma::vec Fitted(const arma::mat& x, const arma::mat& betas)
    {
        return sum(betas % x, 1);
    }

    /**
     * \~english
     * @brief Calculate sum of squared residuals by given \f$X\f$, \f$y\f$, and \f$\beta\f$.
     * 
     * @param x Independent variables \f$X\f$.
     * @param y Dependent variable \f$y\f$.
     * @param betas Coefficient estimates \f$\beta\f$.
     * @return double Sum of squared residuals.
     * 
     * \~chinese
     * @brief 根据给定的 \f$X\f$, \f$y\f$ 和 \f$\beta\f$ 计算残差平方和。
     * 
     * @param x 自变量矩阵 \f$X\f$。
     * @param y 因变量 \f$y\f$。
     * @param betas 回归系数估计值 \f$\beta\f$。
     * @return double 残差平方和。
     * 
     */
    static double RSS(const arma::mat& x, const arma::mat& y, const arma::mat& betas)
    {
        arma::vec r = y - Fitted(x, betas);
        return sum(r % r);
    }

    /**
     * \~english
     * @brief Calculate AICc value according to given \f$X\f$, \f$y\f$, \f$\beta\f$, \f$tr(S)\f$ and \f$tr(SS^T)\f$.
     * 
     * @param x Independent variables \f$X\f$.
     * @param y Dependent variable \f$y\f$.
     * @param betas Coefficient estimates \f$\beta\f$.
     * @param shat A vector of 2 elements: \f$tr(S)\f$ and \f$tr(SS^T)\f$.
     * @return double AICc value.
     * 
     * \~chinese
     * @brief 根据给定的 \f$X\f$, \f$y\f$, \f$\beta\f$, \f$tr(S)\f$ 和 \f$tr(SS^T)\f$ 计算 AICc 值。
     * 
     * @param x 自变量矩阵 \f$X\f$。
     * @param y 因变量 \f$y\f$。
     * @param betas 回归系数估计值 \f$\beta\f$。
     * @param shat 一个包含两个元素的向量，两个元素分别是 \f$tr(S)\f$ 和 \f$tr(SS^T)\f$。
     * @return double AICc 值。
     * 
     */
    static double AICc(const arma::mat& x, const arma::mat& y, const arma::mat& betas, const arma::vec& shat)
    {
        double ss = RSS(x, y, betas), n = (double)x.n_rows;
        return n * log(ss / n) + n * log(2 * arma::datum::pi) + n * ((n + shat(0)) / (n - 2 - shat(0)));
    }

public:

    /**
     * \~english
     * @brief Construct a new CGwmGWRBase object.
     * 
     * \~chinese
     * @brief 构造 CGwmGWRBase 对象。
     * 
     */
    GWRBase() {}

    /**
     * \~english
     * @brief Construct a new CGwmGWRBase object.
     * 
     * @param x Independent variables.
     * @param y Dependent variable.
     * @param spatialWeight Spatial weighting configuration.
     * @param coords Coordinate matrix.
     * 
     * \~chinese
     * @brief 构造 CGwmGWRBase 对象。
     * 
     * @param x 自变量矩阵。
     * @param y 因变量。
     * @param spatialWeight 空间权重配置。
     * @param coords 坐标矩阵。
     * 
     */
    GWRBase(const arma::mat& x, const arma::vec& y, const SpatialWeight& spatialWeight, const arma::mat& coords) : SpatialMonoscaleAlgorithm(spatialWeight, coords)
    {
        mX = x;
        mY = y;
    }

    /**
     * \~english
     * @brief Destroy the CGwmGWRBase object.
     * 
     * \~chinese
     * @brief 析构 CGwmGWRBase 对象。
     * 
     */
    ~GWRBase() {}

public:

    /**
     * \~english
     * @brief Get coefficient estimates.
     * 
     * @return arma::mat Coefficient estimates.
     * 
     * \~chinese
     * @brief 获取回归系数估计值。
     * 
     * @return arma::mat 回归系数估计值。
     * 
     */
    arma::mat betas() const { return mBetas; }

public:     // Implement IRegressionAnalysis
    virtual arma::vec dependentVariable() const override { return mY; }
    virtual void setDependentVariable(const arma::vec& y) override { mY = y; }

    virtual arma::mat independentVariables() const override { return mX; }
    virtual void setIndependentVariables(const arma::mat& x) override { mX = x; }

    virtual bool hasIntercept() const override { return mHasIntercept; }
    virtual void setHasIntercept(const bool has) override { mHasIntercept = has; }

    virtual RegressionDiagnostic diagnostic() const override { return mDiagnostic; }

public:
    virtual bool isValid() override;

protected:

    arma::mat mX;   //!< \~english Independent variables \f$X\f$ \~chinese 自变量 \f$X\f$
    arma::vec mY;   //!< \~english Dependent varialbe \f$y\f$ \~chinese 因变量 \f$y\f$
    arma::mat mBetas;   //!< \~english Coefficient estimates \f$\beta\f$ \~chinese 回归系数估计值 \f$\beta\f$
    bool mHasIntercept = true;  //!< \~english Indicator of whether has intercept or not  \~chinese 指示是否具有截距

    RegressionDiagnostic mDiagnostic;   //!< \~english Diagnostic information \~chinese 诊断信息
};

}

#endif  // GWRBASE_H