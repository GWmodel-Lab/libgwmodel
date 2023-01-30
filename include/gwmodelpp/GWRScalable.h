#ifndef GWRSCALABLE_H
#define GWRSCALABLE_H

#include <utility>
#include <string>
#include <initializer_list>
#include "GWRBase.h"
#include "RegressionDiagnostic.h"
#include "IBandwidthSelectable.h"
#include "IVarialbeSelectable.h"
#include "IParallelizable.h"
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>


namespace gwm
{

/**
 * @brief \~english Scalable GWR \~chinese 大规模地理加权回归模型
 * 
 */
class GWRScalable : public GWRBase
{
public:

    /**
     * @brief \~english Type of bandwidth criterion. \~chinese 带宽优选指标值类型。
     */
    enum BandwidthSelectionCriterionType
    {
        AIC,    //!< AIC
        CV      //!< CV
    };

    /**
     * @brief \~english Additional parameters for leave-one-out CV. \~chinese 去一十字交叉验证算法附加参数
     */
    struct LoocvParams
    {
        const arma::mat* x; //!< \~english Independent variables \~chinese 自变量指针
        const arma::mat* y; //!< \~english Dependent variables \~chinese 因变量指针
        const arma::uword polynomial; //!< \~english The degree of polynomial kernel \~chinese 多项式核的次数
        const arma::mat* Mx0;
        const arma::mat* My0;
    };

    /**
     * @brief \~english Calculate the value of CV criterion. \~chinese 计算CV值
     * 
     * @param target \~english Variables to optimize \~chinese 要优化的变量
     * @param x \~english Independent variables \~chinese 自变量
     * @param y \~english Dependent variables \~chinese 因变量
     * @param poly \~english The degree of polynomial kernel \~chinese 多项式核的次数
     * @param Mx0 \~english \~chinese
     * @param My0 \~english \~chinese
     * @return double \~english Value of CV criterion \~chinese CV值
     */
    static double Loocv(const arma::vec& target, const arma::mat& x, const arma::vec& y, arma::uword poly, const arma::mat& Mx0, const arma::mat& My0);

    /**
     * @brief \~english Calculate the value of AIC criterion. \~chinese 计算AIC值
     * 
     * @param target \~english Variables to optimize \~chinese 要优化的变量
     * @param x \~english Independent variables \~chinese 自变量
     * @param y \~english Dependent variables \~chinese 因变量
     * @param poly \~english The degree of polynomial kernel \~chinese 多项式核的次数
     * @param Mx0 \~english \~chinese
     * @param My0 \~english \~chinese
     * @return double \~english Value of AIC criterion \~chinese AIC值
     */
    static double AICvalue(const arma::vec& target, const arma::mat& x, const arma::vec& y, arma::uword poly, const arma::mat& Mx0, const arma::mat& My0);

private:

    /**
     * @brief \~english Calculate diagnostic information. \~chinese 计算诊断信息。
     * 
     * @param x \~english Independent variables \~chinese 自变量
     * @param y \~english Dependent variables \~chinese 因变量
     * @param betas \~english Coefficient estimates \~chinese 回归系数估计值
     * @param shat \~english A vector of trace of \f$S\f$ and \f$S'S\f$ \~chinese 一个包含 \f$S\f$ 和 \f$S'S\f$ 矩阵迹的向量
     * @return RegressionDiagnostic \~english Diagnostic information \~chinese 诊断信息
     */
    static RegressionDiagnostic CalcDiagnostic(const arma::mat& x, const arma::vec& y, const arma::mat& betas, const arma::vec& shat);

public:

    /**
     * @brief \~english Construct a new GWRScalable object. \~chinese 构造一个新的 GWRScalable 对象。 
     */
    GWRScalable(){};

    /**
     * @brief \~english Destroy the GWRScalable object. \~chinese 销毁 GWRScalable 对象。 
     */
    ~GWRScalable(){};

    /**
     * @brief \~english Get the degree of polynomial kernel  \~chinese 获取多项式核的次数
     * 
     * @return arma::uword \~english The degree of polynomial kernel \~chinese 多项式核的次数
     */
    arma::uword polynomial() const { return mPolynomial; }

    /**
     * @brief \~english Set the degree of polynomial kernel \~chinese 设置多项式核的次数
     * 
     * @param polynomial \~english The degree of polynomial kernel \~chinese 多项式核的次数
     */
    void setPolynomial(arma::uword polynomial) { mPolynomial = polynomial; }

    /**
     * @brief \~english Get the value of CV criterion \~chinese 获取CV值
     * 
     * @return double \~english Value of CV criterion \~chinese CV值
     */
    double cv() const { return mCV; }

    /**
     * @brief \~english Get the scale. \~chinese 获取 scale 的值。
     * 
     * @return double \~english The value of scale \~chinese scale 的值
     */
    double scale() const { return mScale; }

    /**
     * @brief \~english Get the penalty. \~chinese 获取 penalty 的值。
     * 
     * @return double \~english The value of penalty \~chinese penalty 的值
     */
    double penalty() const { return mPenalty; }

    /**
     * @brief \~english Get whether has hat matrix. \~chinese 获取是否有帽子矩阵。
     * 
     * @return true \~english Yes \~chinese 是
     * @return false \~english No \~chinese 否
     */
    bool hasHatMatrix() const { return mHasHatMatrix; }

    /**
     * @brief \~english Set whether has hat matrix. \~chinese 设置是否有帽子矩阵。 
     * 
     * @param flag \~english Whether has hat matrix. \~chinese 是否有帽子矩阵。 
     */
    void setHasHatMatrix(const bool has) { mHasHatMatrix = has; }

    /**
     * @brief \~english Get the type of calculator for parameter optimization criterion. \~chinese 获取计算优化参数指标值的类型。
     * 
     * @return BandwidthSelectionCriterionType \~english Type of calculator for parameter optimization criterion \~chinese 计算优化参数指标值的函数
     */
    BandwidthSelectionCriterionType parameterOptimizeCriterion() const
    {
        return mParameterOptimizeCriterion;
    }

    /**
     * @brief \~english Set the calculator for parameter optimization criterion. \~chinese 设置计算优化参数指标值的类型。
     * 
     * @param parameterOptimizeCriterion \~english Type of calculator for parameter optimization criterion \~chinese 计算优化参数指标值的函数类型
     */
    void setParameterOptimizeCriterion(const BandwidthSelectionCriterionType &parameterOptimizeCriterion)
    {
        mParameterOptimizeCriterion = parameterOptimizeCriterion;
    }

public:     // SpatialAlgorithm interface
    bool isValid() override;


public:     // IRegressionAnalysis interface
    arma::mat fit() override;

    arma::mat predict(const arma::mat& locations) override;

private:

    /**
     * @brief \~english Find the nearest data points. \~chinese 获取临近的数据点。
     * 
     */
    void findDataPointNeighbours();

    /**
     * @brief \~english Find neighbours. \~chinese 获取近邻点。
     * 
     * @param points \~english Coordinates of points \~chinese 样本点坐标
     * @param nnIndex \~english [out] Indeces of nearest neighbours \~chinese [出参] 近邻点索引值
     * @return arma::mat \~english Distance to nearest neighbours \~chinese 到最近邻点的距离
     */
    arma::mat findNeighbours(const arma::mat& points, arma::umat &nnIndex);

    /**
     * @brief \~english Optimize parameters. \~chinese 优化参数。
     * 
     * @param Mx0 \~english  \~chinese 
     * @param My0 \~english  \~chinese 
     * @param b_tilde [out] \~english  \~chinese 
     * @param alpha [out] \~english  \~chinese 
     * @return double \~english Value of criterion \~chinese 指标值
     */
    double optimize(const arma::mat& Mx0, const arma::mat& My0, double& b_tilde, double& alpha);
    
    /**
     * @brief \~english Prepare matrices. \~chinese 矩阵预处理。
     */
    void prepare();

    /**
     * @brief \~english Non-parallel implementation of fitting function. \~chinese 拟合函数的非并行实现。
     * 
     * @param x \~english Independent variables \~chinese 自变量
     * @param y \~english Dependent variables \~chinese 因变量
     * @return arma::mat \~english Coefficient estimates \~chinese 回归系数估计值
     */
    arma::mat fitSerial(const arma::mat &x, const arma::vec &y);

    /**
     * @brief \~english Non-parallel implementation of prediction function. \~chinese 预测函数的非并行实现。
     * 
     * @param locations \~english Locations to predict \~chinese 要预测的位置
     * @param x \~english Independent variables \~chinese 自变量
     * @param y \~english Dependent variables \~chinese 因变量
     * @return arma::mat \~english Coefficient estimates \~chinese 回归系数估计值
     */
    arma::mat predictSerial(const arma::mat& locations, const arma::mat& x, const arma::vec& y);

private:
    arma::uword mPolynomial = 4;    //!< \~english The degree of polynomial kernel \~chinese 多项式核的次数
    size_t mMaxIter = 500;  //!< \~english Maximum iteration for parameter optimization \~chinese 参数优化过程最大迭代次数
    double mCV = 0.0;       //!< \~english Value of CV criterion \~chinese CV值
    double mScale = 1.0;    //!< \~english Scale \~chinese Scale
    double mPenalty = 0.01; //!< \~english Penalty \~chinese Penalty

    bool mHasHatMatrix = true;  //!< \~english Whether has hat matrix. \~chinese 是否有帽子矩阵。 

    SpatialWeight mDpSpatialWeight; //!< \~english Spatial weighting scheme for data points \~chinese 数据点空间权重配置

    BandwidthSelectionCriterionType mParameterOptimizeCriterion = BandwidthSelectionCriterionType::CV; //!< \~english Type of calculator for parameter optimization criterion \~chinese 计算优化参数指标值的函数类型
    

    arma::mat mG0;
    arma::umat mDpNNIndex;  //!< \~english Indeces of nearest data points \~chinese 近邻数据点索引值
    arma::mat mDpNNDists;   //!< \~english Distance to nearest data points \~chinese 近邻数据点距离
    arma::mat mMx0;
    arma::mat mMxx0;
    arma::mat mMy0;
    arma::vec mShat;    //!< \~english A vector of trace of \f$S\f$ and \f$S'S\f$ \~chinese 一个包含 \f$S\f$ 和 \f$S'S\f$ 矩阵迹的向量
    arma::mat mBetasSE; //!< \~english Standard error of coefficient estimates \~chinese 回归系数估计值标准误差。
};

}

#endif  // GWRSCALABLE_H
