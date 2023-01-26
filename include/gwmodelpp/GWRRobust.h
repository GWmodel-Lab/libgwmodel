#ifndef GWRROBUST_H
#define GWRROBUST_H

#include <utility>
#include <string>
#include <initializer_list>
#include "GWRBase.h"
#include "RegressionDiagnostic.h"
#include "IBandwidthSelectable.h"
#include "IVarialbeSelectable.h"
#include "IParallelizable.h"
#include "GWRBasic.h"

namespace gwm
{

/**
 * @brief \~english Robust GWR \~chinese 鲁棒地理加权回归模型
 * 
 */
class GWRRobust : public GWRBasic
{
private:
    typedef arma::mat (GWRRobust::*RegressionHatmatrix)(const arma::mat &, const arma::vec &, arma::mat &, arma::vec &, arma::vec &, arma::mat &); //!< \~english Calculator for fitting \~chinese 拟合函数

    /**
     * @brief \~english Calculate diagnostic information. \~chinese 计算诊断信息。
     * 
     * @param x \~english Independent variables \~chinese 自变量
     * @param y \~english Dependent variables \~chinese 因变量
     * @param betas \~english Coefficient estimates \~chinese 回归系数估计值
     * @param shat \~english A vector of trace of \f$S\f$ and \f$S'S\f$ \~chinese 一个包含 \f$S\f$ 和 \f$S'S\f$ 矩阵迹的向量
     * @return RegressionDiagnostic \~english Diagnostic information \~chinese 诊断信息
     */
    static RegressionDiagnostic CalcDiagnostic(const arma::mat &x, const arma::vec &y, const arma::mat &betas, const arma::vec &shat);

public:

    /**
     * @brief \~english Construct a new GWRRobust object. \~chinese 构造一个新的 GWRRobust 对象。 
     * 
     */
    GWRRobust() {}

    /**
     * @brief \~english Destroy the GWRRobust object. \~chinese 销毁 GWRRobust 对象。 
     * 
     */
    ~GWRRobust() {}

public:

    /**
     * @brief \~english Get whether to use filtered algorithm \~chinese 获取是否使用 Filtered 算法
     * 
     * @return true \~english Yes \~chinese 是
     * @return false \~english No \~chinese 否
     */
    bool filtered() const { return mFiltered; }

    /**
     * @brief \~english Set whether to use filtered algorithm \~chinese 设置是否使用 Filtered 算法
     * 
     * @param value \~english Whether to use filtered algorithm \~chinese 是否使用 Filtered 算法
     */
    void setFiltered(bool value) { mFiltered = value; }

public: // Implement IRegressionAnalysis
    arma::mat predict(const arma::mat& locations) override;
    arma::mat fit() override;
    arma::mat regressionHatmatrix(const arma::mat &x, const arma::vec &y, arma::mat &betasSE, arma::vec &shat, arma::vec &qdiag, arma::mat &S);

private:
    arma::mat fitSerial(const arma::mat& x, const arma::vec& y, arma::mat& betasSE, arma::vec& shat, arma::vec& qDiag, arma::mat& S);
       
#ifdef ENABLE_OPENMP
    //arma::mat predictOmp(const arma::mat& locations, const arma::mat& x, const arma::vec& y);
    arma::mat fitOmp(const arma::mat& x, const arma::vec& y, arma::mat& betasSE, arma::vec& shat, arma::vec& qDiag, arma::mat& S);
#endif

protected:

    /**
     * @brief \~english First type of calibration algorithm. \~chinese 第一种解法。
     * 
     * @param x \~english Independent variables \~chinese 自变量
     * @param y \~english Dependent variables \~chinese 因变量
     * @param betasSE \~english [out] Standard error of coefficient estimates \~chinese [出参] 回归系数估计值标准误差。
     * @param shat \~english [out] A vector of trace of \f$S\f$ and \f$S'S\f$ \~chinese [出参]  一个包含 \f$S\f$ 和 \f$S'S\f$ 矩阵迹的向量
     * @param qdiag \~english [out] \~english Diagonal elements of matrix \f$Q\f$ \~chinese [出参] 矩阵 \f$Q\f$ 的对角线元素
     * @param S \~english [out] Hat matrix \f$S\f$ \~chinese [出参] 帽子矩阵 \f$S\f$
     * @return arma::mat \~english Coefficient estimates \~chinese 回归系数估计值
     */
    arma::mat robustGWRCaliFirst(const arma::mat &x, const arma::vec &y, arma::mat &betasSE, arma::vec &shat, arma::vec &qDiag, arma::mat &S);
    
    /**
     * @brief \~english Second type of calibration algorithm. \~chinese 第二种解法。
     * 
     * @param x \~english Independent variables \~chinese 自变量
     * @param y \~english Dependent variables \~chinese 因变量
     * @param betasSE \~english [out] Standard error of coefficient estimates \~chinese [出参] 回归系数估计值标准误差。
     * @param shat \~english [out] A vector of trace of \f$S\f$ and \f$S'S\f$ \~chinese [出参]  一个包含 \f$S\f$ 和 \f$S'S\f$ 矩阵迹的向量
     * @param qdiag \~english [out] \~english Diagonal elements of matrix \f$Q\f$ \~chinese [出参] 矩阵 \f$Q\f$ 的对角线元素
     * @param S \~english [out] Hat matrix \f$S\f$ \~chinese [出参] 帽子矩阵 \f$S\f$
     * @return arma::mat \~english Coefficient estimates \~chinese 回归系数估计值
     */
    arma::mat robustGWRCaliSecond(const arma::mat &x, const arma::vec &y, arma::mat &betasSE, arma::vec &shat, arma::vec &qDiag, arma::mat &S);
    
    /**
     * @brief \~english Calculate the second-level weights. \~chinese 计算二次权重函数。
     * 
     * @param residual \~english Residuals \~chinese 残差
     * @param mse \~english Mean of squared error \~chinese 平方残差平均数
     * @return arma::vec \~english Second-level weights \~chinese 二次加权权重
     */
    arma::vec filtWeight(arma::vec residual, double mse);

public : // Implement IParallelizable
    void setParallelType(const ParallelType &type) override;

protected:

    /**
     * @brief  \~english Create prediction distance parameter. \~chinese 构造用于预测的距离参数。 
     * 
     * @param locations \~english Locations where to predict coefficients \~chinese 要预测回归系数的位置
     */
    void createPredictionDistanceParameter(const arma::mat& locations);

private:
    bool mFiltered; //!< \~english Whether to use filtered algorithm \~chinese 是否使用 Filtered 算法

    arma::mat mS;           //!< \~english Hat matrix \f$S\f$ \~chinese 帽子矩阵 \f$S\f$
    arma::vec mWeightMask;  //!< \~english Second-level weights \~chinese 二次加权权重
    
    RegressionHatmatrix mfitFunction = &GWRRobust::fitSerial;   //!< \~english Calculator for fitting \~chinese 拟合函数
};

}

#endif // GWRROBUST_H