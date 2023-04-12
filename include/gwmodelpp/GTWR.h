#ifndef GTWR_H
#define GTWR_H

#include <utility>
#include <string>
#include <initializer_list>
#include "GWRBase.h"
#include "RegressionDiagnostic.h"
#include "IBandwidthSelectable.h"
#include "IVarialbeSelectable.h"
#include "IParallelizable.h"
#include "spatialweight/CRSSTDistance.h"

namespace gwm
{

/**
 * \~english
 * @brief Basic implementation of geographically temporally weighted regression.
 * This algorithm can auto select bandwidth.
 * 
 * \~chinese
 * @brief 时空地理加权回归算法的实现。
 * 该算法可以自动选带宽。
 * 
 */
class GTWR :  public GWRBase, public IBandwidthSelectable, public IParallelizable, public IParallelOpenmpEnabled
{
public:

    /**
     * \~english
     * @brief Type of criterion for bandwidth selection.
     * \~chinese
     * @brief 用于带宽优选的指标类型。
     */
    enum BandwidthSelectionCriterionType
    {
        AIC,
        CV
    };

    static std::unordered_map<BandwidthSelectionCriterionType, std::string> BandwidthSelectionCriterionTypeNameMapper;
    
    typedef arma::mat (GTWR::*PredictCalculator)(const arma::mat&, const arma::mat&, const arma::vec&);//!< \~english Predict function declaration. \~chinese 预测函数声明。
    typedef arma::mat (GTWR::*FitCalculator)(const arma::mat&, const arma::vec&, arma::mat&, arma::vec&, arma::vec&, arma::mat&);//!< \~english Fit function declaration. \~chinese 拟合函数声明。

    typedef double (GTWR::*BandwidthSelectionCriterionCalculator)(BandwidthWeight*);//!< \~english Declaration of criterion calculator for bandwidth selection. \~chinese 带宽优选指标计算函数声明。
    typedef double (GTWR::*IndepVarsSelectCriterionCalculator)(const std::vector<std::size_t>&);//!< \~english Declaration of criterion calculator for variable selection. \~chinese 变量优选指标计算函数声明。

private:

    /**
     * \~english
     * @brief Calculate diagnostic information.
     * 
     * @param x Independent variables \f$X\f$.
     * @param y Dependent variable \f$y\f$.
     * @param betas Coefficient estimates \f$\beta\f$.
     * @param shat A vector of 2 elements: \f$tr(S)\f$ and \f$tr(SS^T)\f$.
     * @return GwmRegressionDiagnostic Diagnostic information.
     * 
     * \~chinese
     * @brief 计算诊断信息。
     * 
     * @param x 自变量矩阵 \f$X\f$。
     * @param y 因变量 \f$y\f$。
     * @param betas 回归系数估计值 \f$\beta\f$。
     * @param shat 一个包含两个元素的向量，两个元素分别是 \f$tr(S)\f$ 和 \f$tr(SS^T)\f$。
     * @return GwmRegressionDiagnostic 诊断信息。
     * 
     */
    static RegressionDiagnostic CalcDiagnostic(const arma::mat& x, const arma::vec& y, const arma::mat& betas, const arma::vec& shat);

public:

    /**
     * \~english
     * @brief Construct a new GTWR object.
     * \~chinese
     * @brief 构造 GTWR 对象。
     */
    GTWR(){};

    /**
     * \~english
     * @brief Construct a new GTWR object.
     * 
     * @param x Independent variables.
     * @param y Dependent variable.
     * @param coords Coordinate matrix.
     * @param spatialWeight Spatial weighting configuration.
     * @param hasHatMatrix Whether has hat-matrix.
     * @param hasIntercept Whether has intercept.
     * 
     * \~chinese
     * @brief 构造 GTWR 对象。
     * 
     * @param x 自变量矩阵。
     * @param y 因变量。
     * @param coords 坐标矩阵。
     * @param spatialWeight 空间权重配置。
     * @param hasHatMatrix 是否计算帽子矩阵。
     * @param hasIntercept 是否有截距。
     */
    GTWR(const arma::mat& x, const arma::vec& y, const arma::mat& coords, const SpatialWeight& spatialWeight, bool hasHatMatrix = true, bool hasIntercept = true)
        : GWRBase(x, y, spatialWeight, coords)
    {
        mHasHatMatrix = hasHatMatrix;
        mHasIntercept = hasIntercept;
    }

    /**
     * \~english
     * @brief Destroy the GTWR object.
     * \~chinese
     * @brief 析构 GTWR 对象。
     */
    ~GTWR(){};

private:
    Weight* mWeight = nullptr;      //!< \~english weight pointer. \~chinese 权重指针。
    Distance* mDistance = nullptr;  //!< \~english distance pointer. \~chinese 距离指针。

//public:
//    arma::vec weightVector(uword focus);//recalculate weight using spatial temporal distance

public:

    /**
     * \~english
     * @brief Get whether auto select bandwidth.
     * 
     * @return true if auto select bandwidth.
     * @return false if not auto select bandwidth.
     * 
     * \~chinese
     * @brief 获取是否自动优选带宽。
     * 
     * @return true 如果自动优选带宽。
     * @return false 如果不自动优选带宽。
     * 
     */
    bool isAutoselectBandwidth() const { return mIsAutoselectBandwidth; }
  
    /**
     * \~english
     * @brief Set whether auto select bandwidth.
     * 
     * @param isAutoSelect true if auto select bandwidth, otherwise false.
     * 
     * \~chinese
     * @brief 设置是否自动优选带宽。
     * 
     * @param isAutoSelect true 如果要自动优选带宽，否则 false。
     */
    void setIsAutoselectBandwidth(bool isAutoSelect) { mIsAutoselectBandwidth = isAutoSelect; }

    /**
     * \~english
     * @brief Get type of criterion for bandwidth selection.
     * 
     * @return BandwidthSelectionCriterionType Type of criterion for bandwidth selection.
     * 
     * \~chinese
     * @brief 获取带宽自动优选指标值类型。
     * 
     * @return BandwidthSelectionCriterionType 带宽自动优选指标值类型。
     */
    BandwidthSelectionCriterionType bandwidthSelectionCriterion() const { return mBandwidthSelectionCriterion; }
    
    /**
     * \~english
     * @brief Set type of criterion for bandwidth selection.
     * 
     * @param criterion Type of criterion for bandwidth selection.
     * 
     * \~chinese
     * @brief 设置带宽自动优选指标值类型。
     * 
     * @param criterion 带宽自动优选指标值类型。
     */
    void setBandwidthSelectionCriterion(const BandwidthSelectionCriterionType& criterion);

    /**
     * \~english
     * @brief Get criterion list for bandwidth selection.
     * 
     * @return BandwidthCriterionList Criterion list for bandwidth selection.
     * 
     * \~chinese
     * @brief 获取带宽优选过程的指标值列表。
     * 
     * @return BandwidthCriterionList 带宽优选过程的指标值列表。
     */
    BandwidthCriterionList bandwidthSelectionCriterionList() const { return mBandwidthSelectionCriterionList; }

    /**
     * \~english
     * @brief Get whether has hat-matrix.
     * 
     * @return true if has hat-matrix.
     * @return false if does not have hat-matrix.
     * 
     * \~chinese
     * @brief 获取是否计算帽子矩阵。
     * 
     * @return true 如果计算帽子矩阵。
     * @return false 如果不计算帽子矩阵。
     */
    bool hasHatMatrix() const { return mHasHatMatrix; }

    /**
     * \~english
     * @brief Set the Has Hat-Matrix object
     * 
     * @param has true if has hat-matrix, otherwise false.
     * 
     * \~chinese
     * @brief 设置是否计算帽子矩阵。
     * 
     * @param has true 如果计算帽子矩阵，否则 false。
     */
    void setHasHatMatrix(const bool has) { mHasHatMatrix = has; }

    // void setTimes(const arma::vec& times)
    // {
    //     vTimes=times;
    // }

    /**
     * \~english
     * @brief Set the spatial and temporal data.
     * 
     * @param coords spatial coordinates mat.
     * @param times temporal stamps vector.
     * 
     * 
     * \~chinese
     * @brief 设置空间坐标和时间向量。
     * 
     * @param coords 空间坐标矩阵.
     * @param times 时间戳向量.
     * 
     */
    void setCoords(const arma::mat& coords, const arma::vec& times)
    {
        mCoords=coords;
        vTimes=times;
    }

    /**
     * \~english
     * @brief Get standard errors of coefficient estimates.
     * 
     * @return arma::mat Standard errors of coefficient estimates.
     * 
     * \~chinese
     * @brief 获取回归系数估计值的标准差。
     * 
     * @return arma::mat 回归系数估计值的标准差。
     */
    arma::mat betasSE() { return mBetasSE; }
    
    /**
     * \~english
     * @brief Get a vector of \f$tr(S)\f$ and \f$tr(SS^T)\f$.
     * 
     * @return arma::vec A vector of \f$tr(S)\f$ and \f$tr(SS^T)\f$.
     * 
     * \~chinese
     * @brief 获取一个由 \f$tr(S)\f$ 和 \f$tr(SS^T)\f$ 组成的向量。
     * 
     * @return arma::vec 由 \f$tr(S)\f$ 和 \f$tr(SS^T)\f$ 组成的向量。
     */
    arma::vec sHat() { return mSHat; }
    
    /**
     * \~english
     * @brief Get the diagonal elements of matrix \f$Q\f$.
     * 
     * @return arma::vec The diagonal elements of matrix \f$Q\f$.
     * 
     * \~chinese
     * @brief 获取矩阵 \f$Q\f$ 的对角线元素。
     * 
     * @return arma::vec 矩阵 \f$Q\f$ 的对角线元素。
     */
    arma::vec qDiag() { return mQDiag; }
    
    /**
     * \~english
     * @brief Get the hat-matrix \f$S\f$.
     * 
     * @return arma::mat The hat-matrix \f$S\f$.
     * 
     * \~chinese
     * @brief 获取帽子矩阵 \f$S\f$。
     * 
     * @return arma::mat 帽子矩阵 \f$S\f$。
     */
    arma::mat s() { return mS; }

public:     // Implement Algorithm
    bool isValid() override;

public:     // Implement IRegressionAnalysis
    arma::mat predict(const arma::mat& locations) override;

    arma::mat fit() override;

private:

    /**
     * \~english 
     * @brief Predict coefficients on specified locations (serial implementation).
     * 
     * @param locations Locations where to predict coefficients.
     * @param x Independent variables.
     * @param y Dependent variable.
     * @return mat Predicted coefficients.
     * 
     * \~chinese 
     * @brief 在指定位置处进行回归系数预测（单线程实现）。
     * 
     * @param locations 指定位置。
     * @param x 自变量矩阵。
     * @param y 因变量。
     * @return mat 回归系数预测值。
     */
    arma::mat predictSerial(const arma::mat& locations, const arma::mat& x, const arma::vec& y);
        
    /**
     * \~english
     * @brief Fit coefficients (serial implementation).
     * 
     * @param x Independent variables.
     * @param y Dependent variable.
     * @param betasSE [out] Standard errors of coefficient estimates.
     * @param shat [out] A vector of \f$tr(S)\f$ and \f$tr(SS^T)\f$.
     * @param qDiag [out] The diagonal elements of matrix \f$Q\f$.
     * @param S [out] The hat-matrix \f$S\f$.
     * @return mat Coefficient estimates.
     * 
     * \~chinese
     * @brief 回归系数估计值（串行实现）。
     * 
     * @param x 自变量矩阵。
     * @param y 因变量。
     * @param betasSE [out] 回归系数估计值的标准差。
     * @param shat [out] 由 \f$tr(S)\f$ 和 \f$tr(SS^T)\f$ 组成的向量。
     * @param qDiag [out] 矩阵 \f$Q\f$ 的对角线元素。
     * @param S [out] 帽子矩阵 \f$S\f$。
     * @return mat 回归系数估计值
     */
    arma::mat fitSerial(const arma::mat& x, const arma::vec& y, arma::mat& betasSE, arma::vec& shat, arma::vec& qDiag, arma::mat& S);
    
#ifdef ENABLE_OPENMP
    /**
     * \~english 
     * @brief Predict coefficients on specified locations (OpenMP implementation).
     * 
     * @param locations Locations where to predict coefficients.
     * @param x Independent variables.
     * @param y Dependent variable.
     * @return mat Predicted coefficients.
     * 
     * \~chinese 
     * @brief 在指定位置处进行回归系数预测（OpenMP 实现）。
     * 
     * @param locations 指定位置。
     * @param x 自变量矩阵。
     * @param y 因变量。
     * @return mat 回归系数预测值。
     */
    arma::mat predictOmp(const arma::mat& locations, const arma::mat& x, const arma::vec& y);
    
    /**
     * \~english
     * @brief Fit coefficients (OpenMP implementation).
     * 
     * @param x Independent variables.
     * @param y Dependent variable.
     * @param betasSE [out] Standard errors of coefficient estimates.
     * @param shat [out] A vector of \f$tr(S)\f$ and \f$tr(SS^T)\f$.
     * @param qDiag [out] The diagonal elements of matrix \f$Q\f$.
     * @param S [out] The hat-matrix \f$S\f$.
     * @return mat Coefficient estimates.
     * 
     * \~chinese
     * @brief 回归系数估计值（OpenMP 实现）。
     * 
     * @param x 自变量矩阵。
     * @param y 因变量。
     * @param betasSE [out] 回归系数估计值的标准差。
     * @param shat [out] 由 \f$tr(S)\f$ 和 \f$tr(SS^T)\f$ 组成的向量。
     * @param qDiag [out] 矩阵 \f$Q\f$ 的对角线元素。
     * @param S [out] 帽子矩阵 \f$S\f$。
     * @return mat 回归系数估计值
     */
    arma::mat fitOmp(const arma::mat& x, const arma::vec& y, arma::mat& betasSE, arma::vec& shat, arma::vec& qDiag, arma::mat& S);
#endif

public:     // Implement IBandwidthSelectable
    double getCriterion(BandwidthWeight* weight) override
    {
        return (this->*mBandwidthSelectionCriterionFunction)(weight);
    }

private:

    /**
     * \~english
     * @brief Get CV value with given bandwidth for bandwidth optimization (serial implementation).
     * 
     * @param weight Given bandwidth
     * @return double Criterion value
     * 
     * \~chinese
     * @brief 根据指定的带宽计算带宽优选的CV值（串行实现）。
     * 
     * @param weight 指定的带宽。
     * @return double 带宽优选的指标值。
     */
    double bandwidthSizeCriterionCVSerial(BandwidthWeight* bandwidthWeight);
        
    /**
     * \~english
     * @brief Get AIC value with given bandwidth for bandwidth optimization (serial implementation).
     * 
     * @param weight Given bandwidth
     * @return double Criterion value
     * 
     * \~chinese
     * @brief 根据指定的带宽计算带宽优选的AIC值（串行实现）。
     * 
     * @param weight 指定的带宽。
     * @return double 带宽优选的指标值。
     */
    double bandwidthSizeCriterionAICSerial(BandwidthWeight* bandwidthWeight);

#ifdef ENABLE_OPENMP
    /**
     * \~english
     * @brief Get CV value with given bandwidth for bandwidth optimization (OpenMP implementation).
     * 
     * @param weight Given bandwidth
     * @return double Criterion value
     * 
     * \~chinese
     * @brief 根据指定的带宽计算带宽优选的CV值（OpenMP 实现）。
     * 
     * @param weight 指定的带宽。
     * @return double 带宽优选的指标值。
     */
    double bandwidthSizeCriterionCVOmp(BandwidthWeight* bandwidthWeight);
    
    /**
     * \~english
     * @brief Get AIC value with given bandwidth for bandwidth optimization (OpenMP implementation).
     * 
     * @param weight Given bandwidth
     * @return double Criterion value
     * 
     * \~chinese
     * @brief 根据指定的带宽计算带宽优选的AIC值（OpenMP 实现）。
     * 
     * @param weight 指定的带宽。
     * @return double 带宽优选的指标值。
     */
    double bandwidthSizeCriterionAICOmp(BandwidthWeight* bandwidthWeight);
#endif


public:     // Implement IParallelizable
    int parallelAbility() const override
    {
        return ParallelType::SerialOnly
#ifdef ENABLE_OPENMP
            | ParallelType::OpenMP
#endif        
        ;
    }

    ParallelType parallelType() const override { return mParallelType; }

    void setParallelType(const ParallelType& type) override;

public:     // Implement IGwmParallelOpenmpEnabled
    void setOmpThreadNum(const int threadNum) override { mOmpThreadNum = threadNum; }

protected:

    /**
     * \~english
     * @brief Whether to store hat-matrix \f$S\f$.
     * 
     * @return true if store hat-matrix.
     * @return false if not to store hat-matrix.
     * 
     * \~chinese
     * @brief 是否保存帽子矩阵 \f$S\f$.
     * 
     * @return true 如果保存帽子矩阵。
     * @return false 如果不保存帽子矩阵。
     * 
     */
    bool isStoreS() { return mHasHatMatrix && (mCoords.n_rows < 8192); }

    /**
     * \~english
     * @brief Create distance parameters for prediction.
     * 
     * @param locations Distance parameters for prediction.
     * 
     * \~chinese
     * @brief 生成用于预测的距离参数。
     * 
     * @param locations 用于预测的距离参数。
     * 
     */
    void createPredictionDistanceParameter(const arma::mat& locations);

    
    /**
     * \~english
     * @brief Create distance parameters.
     * \~chinese
     * @brief 生成距离参数。
     */
    void createDistanceParameter();

    // double LambdaAutoSelection();

protected:

    bool mHasHatMatrix = true;  //!< \~english Whether has hat-matrix. \~chinese 是否具有帽子矩阵。
    bool mHasFTest = false;     //!< @todo \~english Whether has F-test \~chinese 是否具有F检验。
    bool mHasPredict = false;   //!< @deprecated \~english Whether has variables to predict dependent variable. \~chinese 是否有预测位置处的变量。

    bool mIsAutoselectBandwidth = false;//!< \~english Whether need bandwidth autoselect. \~chinese 是否需要自动优选带宽。
    bool mIsAutoselectLambda = false;//!< \~english Whether need lambda autoselect. \~chinese 是否需要自动优选lambda。

    BandwidthSelectionCriterionType mBandwidthSelectionCriterion = BandwidthSelectionCriterionType::AIC;
    BandwidthSelectionCriterionCalculator mBandwidthSelectionCriterionFunction = &GTWR::bandwidthSizeCriterionCVSerial;
    BandwidthCriterionList mBandwidthSelectionCriterionList;

    PredictCalculator mPredictFunction = &GTWR::predictSerial;
    FitCalculator mFitFunction = &GTWR::fitSerial;

    ParallelType mParallelType = ParallelType::SerialOnly; //!< \~english Type of parallel method. \~chinese 并行方法类型。
    int mOmpThreadNum = 8;  //!< \~english Number of threads to create. \~chinese 并行计算创建的线程数。

    arma::mat mBetasSE; //!< \~english Standard errors of coefficient estimates. \~chinese 回归系数估计值的标准差。
    arma::vec mSHat;    //!< \~english A vector of \f$tr(S)\f$ and \f$tr(SS^T)\f$. \~chinese 由 \f$tr(S)\f$ 和 \f$tr(SS^T)\f$ 组成的向量。
    arma::vec mQDiag;   //!< \~english The diagonal elements of matrix \f$Q\f$. \~chinese 矩阵 \f$Q\f$ 的对角线元素。
    arma::mat mS;       //!< \~english The hat-matrix \f$S\f$. \~chinese 帽子矩阵 \f$S\f$。

    arma::vec vTimes;   //!< \~english vectors for timestamp input. \~chinese 输入时间的向量。

};

}

#endif  // GTWR_H