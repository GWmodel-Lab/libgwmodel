#ifndef GWAVERAGE_H
#define GWAVERAGE_H

#include "SpatialMonoscaleAlgorithm.h"
#include "IMultivariableAnalysis.h"
#include "IParallelizable.h"
#include "IBandwidthSelectable.h"

namespace gwm
{

/**
 * \~english
 * @brief The class for Geographically Weighted Summary Statistics. 
 * Geographically Weighted Summary Statistics is an algorithm for calculating local weighted statistics. 
 * They are local mean, local standard deviation, local variance, local skewness, local coefficients of variation, 
 * local covariances, local correlations (Pearson's), local correlations (Spearman's),
 * local medians, local interquartile ranges, local quantile imbalances and coordinates.
 * To get these matrices, call these functions:
 * 
 * - local mean <- GWAverage::localMean()
 * - local standard deviation <- GWAverage::localSDev()
 * - local variance <- GWAverage::localVar()
 * - local skewness <- GWAverage::localSkewness()
 * - local coefficients of variation <- GWAverage::localCV()
 * - local covariances <- GWAverage::localCov()
 * - local correlations (Pearson's) <- GWAverage::localCorr()
 * - local correlations (Spearman's) <- GWAverage::localSCorr()
 * - local medians <- GWAverage::localMedian()
 * - local interquartile ranges <- GWAverage::iqr()
 * - local quantile imbalances and coordinates <- GWAverage::qi()
 * 
 * \~chinese
 * @brief 地理加权汇总统计分析算法类。
 * 地理加权汇总统计是计算局部加权统计值的方法。
 * 可计算的统计值包括： local mean, local standard deviation, local variance, local skewness, local coefficients of variation, 
 * local covariances, local correlations (Pearson's), local correlations (Spearman's),
 * local medians, local interquartile ranges, local quantile imbalances and coordinates.
 * 使用下面这些函数获取上述值：
 * 
 * - local mean <- GWAverage::localMean()
 * - local standard deviation <- GWAverage::localSDev()
 * - local variance <- GWAverage::localVar()
 * - local skewness <- GWAverage::localSkewness()
 * - local coefficients of variation <- GWAverage::localCV()
 * - local covariances <- GWAverage::localCov()
 * - local correlations (Pearson's) <- GWAverage::localCorr()
 * - local correlations (Spearman's) <- GWAverage::localSCorr()
 * - local medians <- GWAverage::localMedian()
 * - local interquartile ranges <- GWAverage::iqr()
 * - local quantile imbalances and coordinates <- GWAverage::qi()
 */
class GWAverage : public SpatialMonoscaleAlgorithm, public IMultivariableAnalysis, public IParallelizable, public IParallelOpenmpEnabled, public IBandwidthSelectable
{
public:

    /**
     * @brief \~english Calculate weighted covariances for two matrices. \~chinese 计算两个矩阵的加权协方差。
     * 
     * @param x1 \~english Matrix \f$ X_1 \f$ \~chinese 矩阵 \f$ X_1 \f$
     * @param x2 \~english Matrix \f$ X_2 \f$ \~chinese 矩阵 \f$ X_2 \f$
     * @param w \~english Weight vector \f$ w \f$ \~chinese 权重向量 \f$ w \f$
     * @return \~english Weighted covariances \f[ cov(X_1,X_2) = \frac{\sum_{i=1}^n w_i(x_{1i} - \bar{x}_1) \sum_{i=1}^n w_i(x_{2i} - \bar{x}_2)}{1 - \sum_{i=1}^n w_i} \f]
     * \~chinese 加权协方差 \f[ cov(X_1,X_2) = \frac{\sum_{i=1}^n w_i(x_{1i} - \bar{x}_1) \sum_{i=1}^n w_i(x_{2i} - \bar{x}_2)}{1 - \sum_{i=1}^n w_i} \f]
     */
    static double covwt(const arma::mat &x1, const arma::mat &x2, const arma::vec &w)
    {
        return sum((sqrt(w) % (x1 - sum(x1 % w))) % (sqrt(w) % (x2 - sum(x2 % w)))) / (1 - sum(w % w));
    }

    /**
     * @brief \~english Calculate weighted correlation for two matrices. \~chinese 计算两个矩阵的加权相关系数。
     * 
     * @param x1 \~english Matrix \f$ X_1 \f$ \~chinese 矩阵 \f$ X_1 \f$
     * @param x2 \~english Matrix \f$ X_2 \f$ \~chinese 矩阵 \f$ X_2 \f$
     * @param w \~english Weight vector \f$ w \f$ \~chinese 权重向量 \f$ w \f$
     * @return \~english Weighted correlation \f[ corr(X_1,X_2) = \frac{cov(X_1,X_2)}{\sqrt{cov(X_1,X_1) cov(X_2,X_2)}} \f]
     * \~english 加权相关系数 \f[ corr(X_1,X_2) = \frac{cov(X_1,X_2)}{\sqrt{cov(X_1,X_1) cov(X_2,X_2)}} \f]
     */
    static double corwt(const arma::mat &x1, const arma::mat &x2, const arma::vec &w)
    {
        return covwt(x1,x2,w)/sqrt(covwt(x1,x1,w)*covwt(x2,x2,w));
    }

    static arma::vec del(arma::vec x, arma::uword rowcount);

    static arma::vec rank(arma::vec x)
    {
        arma::vec n = arma::linspace(0.0, (double)x.n_rows - 1, x.n_rows);
        arma::vec res = n(sort_index(x));
        return n(sort_index(res)) + 1.0;
    }

    typedef void (GWAverage::*SummaryCalculator)();  //!< \~english Calculator for summary statistics \~chinese 汇总统计计算函数

    /**
     * @brief \~english GWAverage working mode. \~chinese GWAverage 工作模式。
     */
    enum class GWAverageMode {
        Average,        //!< \~english Average mode (for one variable) \~chinese 均值模式（针对单变量）
        Correlation     //!< \~english Correlation mode (for variable pairs) \~chinese 相关模式（针对双变量）
    };

protected:
    static arma::vec findq(const arma::mat& x, const arma::vec& w);

public:
    
    /**
     * @brief \~english Construct a new GWAverage object. \~chinese 构造一个新的 GWAverage 对象。
     * 
     */
    GWAverage() {}
    
    /**
     * @brief \~english Construct a new GWAverage object. \~chinese 构造一个新的 GWAverage 对象。
     * 
     */
    GWAverage(const arma::mat x, const arma::mat coords, const SpatialWeight& spatialWeight)
        : SpatialMonoscaleAlgorithm(spatialWeight, coords)
    {
        mX = x;
    }

    /**
     * @brief \~english Destroy the GWAverage object. \~chinese 销毁 GWAverage 对象。
     * 
     */
    ~GWAverage() {}

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

    /**
     * \~english
     * @brief Declaration of criterion calculator for bandwidth selection. 
     * \~chinese
     * @brief 带宽优选指标计算函数声明。
     */
    typedef double (GWAverage::*BandwidthSelectionCriterionCalculator)(BandwidthWeight*);


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
    bool isAutoselectBandwidth() const { return mGWAverageMode==GWAverageMode::Correlation ? mIsAutoselectBandwidth : false; }
  
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

private:

    /**
     * \~english
     * @brief Get CV value with given bandwidth for bandwidth optimization (serial implementation).
     * 
     * @param bandwidthWeight Given bandwidth
     * @return double Criterion value
     * 
     * \~chinese
     * @brief 根据指定的带宽计算带宽优选的CV值（串行实现）。
     * 
     * @param bandwidthWeight 指定的带宽。
     * @return double 带宽优选的指标值。
     */
    double bandwidthSizeCriterionCVSerial(BandwidthWeight* bandwidthWeight);
        
    /**
     * \~english
     * @brief Get AIC value with given bandwidth for bandwidth optimization (serial implementation).
     * 
     * @param bandwidthWeight Given bandwidth
     * @return double Criterion value
     * 
     * \~chinese
     * @brief 根据指定的带宽计算带宽优选的AIC值（串行实现）。
     * 
     * @param weigbandwidthWeightht 指定的带宽。
     * @return double 带宽优选的指标值。
     */
    double bandwidthSizeCriterionAICSerial(BandwidthWeight* bandwidthWeight);

public:     // Implement IBandwidthSelectable
    Status getCriterion(BandwidthWeight* weight, double& criterion) override
    {
        criterion = (this->*mBandwidthSelectionCriterionFunction)(weight);
        return mStatus;
    }

    /**
     * @brief \~english Get whether use quantile algorithms. \~chinese 获取是否使用基于排序的算法。
     * 
     * @return true \~english if use quantile algorithms \~chinese 使用基于排序的算法
     * @return false \~english if not to use quantile algorithms \~chinese 不使用基于排序的算法
     */
    bool quantile() const { return mQuantile; }

    /**
     * @brief \~english Get whether use quantile algorithms. \~chinese 设置是否使用基于排序的算法
     * 
     * @param quantile \~english Whether use quantile algorithms \~chinese 是否使用基于排序的算法
     */
    void setQuantile(bool quantile) { mQuantile = quantile; }

    /**
     * @brief \~english Get whether calculate correlation between the first variable and others. \~chinese 获取是否仅为第一个变量计算与其他变量的相关系数
     * 
     * @return true \~english Yes \~chinese 是
     * @return false \~english No \~chinese 否
     */
    bool isCorrWithFirstOnly() const { return mIsCorrWithFirstOnly; }

    /**
     * @brief \~english Set whether calculate correlation between the first variable and others. \~chinese 设置是否仅为第一个变量计算与其他变量的相关系数 
     * 
     * @param corrWithFirstOnly \~english Whether calculate correlation between the first variable and others. \~chinese 是否仅为第一个变量计算与其他变量的相关系数
     */
    void setIsCorrWithFirstOnly(bool corrWithFirstOnly) { mIsCorrWithFirstOnly = corrWithFirstOnly; }

    /**
     * @brief \~english Get local mean on each sample. \~chinese 获取每个样本的局部均值。
     * 
     * @return \~english Local mean on each sample \~chinese 每个样本的局部均值
     */
    const arma::mat& localMean() const { return mLocalMean; }
    
    /**
     * @brief \~english Get local standard deviation on each sample. \~chinese 获取每个样本的局部标准差。
     * 
     * @return \~english Local standard deviation on each sample \~chinese 每个样本的局部标准差
     */
    const arma::mat& localSDev() const { return mStandardDev; }
    
    /**
     * @brief \~english Get local skewness on each sample. \~chinese 获取每个样本的局部偏度。
     * 
     * @return \~english Local skewness on each sample \~chinese 每个样本的局部偏度
     */
    const arma::mat& localSkewness() const { return mLocalSkewness; }
    
    /**
     * @brief \~english Get local coefficients of variation on each sample. \~chinese 获取每个样本的局部变化系数。
     * 
     * @return \~english Local coefficients of variation on each sample \~chinese 每个样本的局部变化系数
     */
    const arma::mat& localCV() const { return mLCV; }
    
    /**
     * @brief \~english Get local variance on each sample. \~chinese 获取每个样本的局部方差。
     * 
     * @return \~english Local variance on each sample \~chinese 每个样本的局部方差
     */
    const arma::mat& localVar() const { return mLVar; }

    
    /**
     * @brief \~english Get local median on each sample. \~chinese 获取每个样本的局部中位数。
     * 
     * @return \~english Local median on each sample \~chinese 每个样本的局部中位数
     */
    const arma::mat& localMedian() const { return mLocalMedian; }
    
    /**
     * @brief \~english Get local interquartile ranges on each sample. \~chinese 获取每个样本的局部四分位距。
     * 
     * @return \~english Local interquartile ranges on each sample \~chinese 每个样本的局部四分位距
     */
    const arma::mat& iqr() const { return mIQR; }
    
    /**
     * @brief \~english Get local quantile imbalances and coordinates on each sample. \~chinese 获取每个样本的局部分位数不平衡度。
     * 
     * @return \~english Local quantile imbalances and coordinates on each sample \~chinese 每个样本的局部分位数不平衡度
     */
    const arma::mat& qi() const { return mQI; }

    
    /**
     * @brief \~english Get local coefficients of variation on each sample. \~chinese 获取局部协方差。
     * 
     * @return \~english Local coefficients of variation on each sample.
     * If corrWithFirstOnly is set true, the number of columns is the (number of fields) - 1;
     * if not, the number of columns is the (((number of fields) - 1) * (number of fields)) / 2.
     * For variables \f$v_1, v_2, v_3, ... , v_{k-1}, v_k\f$, the fields are arranged as: 
     * \f$cov(v_1,v_2), cov(v_1,v_3), ... , cov(v_1,v_k), cov(v_2,v_3), ... , cov(v_2,v_k), ... , cov(v_{k-1},vk)\f$
     * \~chinese 局部协方差。
     * 如果 corrWithFirstOnly 设置为 true ，则共有 字段数 - 1 列；
     * 否则，有 ((字段数 - 1) * 字段数) / 2 列。
     * 对于变量 \f$v_1, v_2, v_3, ... , v_{k-1}, v_k\f$ 返回字段按如下方式排序：
     * \f$cov(v_1,v_2), cov(v_1,v_3), ... , cov(v_1,v_k), cov(v_2,v_3), ... , cov(v_2,v_k), ... , cov(v_{k-1},vk)\f$
     */
    const arma::mat& localCov() const { return mCovmat; }
    
    /**
     * @brief \~english Get local correlations (Pearson's) on each sample. \~chinese 获取局部皮尔逊相关系数。
     * 
     * @return \~english Local correlations (Pearson's) on each sample.
     * If corrWithFirstOnly is set true, the number of columns is the (number of fields) - 1;
     * if not, the number of columns is the (((number of fields) - 1) * (number of fields)) / 2.
     * For variables \f$v_1, v_2, v_3, ... , v_{k-1}, v_k\f$, the fields are arranged as: 
     * \f$corr(v_1,v_2), corr(v_1,v_3), ... , corr(v_1,v_k), corr(v_2,v_3), ... , corr(v_2,v_k), ... , corr(v_{k-1},vk)\f$
     * \~chinese 局部皮尔逊相关系数。
     * 如果 corrWithFirstOnly 设置为 true ，则共有 字段数 - 1 列；
     * 否则，有 ((字段数 - 1) * 字段数) / 2 列。
     * 对于变量 \f$v_1, v_2, v_3, ... , v_{k-1}, v_k\f$ 返回字段按如下方式排序：
     * \f$corr(v_1,v_2), corr(v_1,v_3), ... , corr(v_1,v_k), corr(v_2,v_3), ... , corr(v_2,v_k), ... , corr(v_{k-1},vk)\f$
     */
    const arma::mat& localCorr() const { return mCorrmat; }
    
    /**
     * @brief \~english Get local correlations (Spearman's) on each sample. \~chinese 获取局部斯皮尔曼相关系数。
     * 
     * @return \~english Local correlations (Spearman's) on each sample.
     * If corrWithFirstOnly is set true, the number of columns is the (number of fields) - 1;
     * if not, the number of columns is the (((number of fields) - 1) * (number of fields)) / 2.
     * For variables \f$v_1, v_2, v_3, ... , v_{k-1}, v_k\f$, the fields are arranged as: 
     * \f$corr(v_1,v_2), corr(v_1,v_3), ... , corr(v_1,v_k), corr(v_2,v_3), ... , corr(v_2,v_k), ... , corr(v_{k-1},vk)\f$
     * \~chinese 局部斯皮尔曼相关系数。
     * 如果 corrWithFirstOnly 设置为 true ，则共有 字段数 - 1 列；
     * 否则，有 ((字段数 - 1) * 字段数) / 2 列。
     * 对于变量 \f$v_1, v_2, v_3, ... , v_{k-1}, v_k\f$ 返回字段按如下方式排序：
     * \f$corr(v_1,v_2), corr(v_1,v_3), ... , corr(v_1,v_k), corr(v_2,v_3), ... , corr(v_2,v_k), ... , corr(v_{k-1},vk)\f$
     */
    const arma::mat& localSCorr() const { return mSCorrmat; }

public:     // SpatialAlgorithm interface
    bool isValid() override;

public:     // IMultivariableAnalysis
    const arma::mat& variables() const override { return mX; }
    
    /**
     * @brief \~english set variables \~chinese 设置变量x。
     * 
     * @param x \~english variables for GWAverage \~chinese 进行GWAverage的变量，如果只有一列，只能进行GWAverage。
     */
    void setVariables(const arma::mat& x) override { mX = x; }


    /**
     * @brief \~english set GWAverage working mode.
     * \~chinese 设置GWAverage的工作模式。
     * 
     * @param mode \~english GWAverage working mode \~chinese GWAverage的工作模式
     */
    void setGWAverageMode(GWAverageMode mode);

    void run() override;

public:     // IParallelizable
    int parallelAbility() const override
    {
        return ParallelType::SerialOnly
#ifdef ENABLE_OPENMP
            | ParallelType::OpenMP
#endif        
            ;
    }
    ParallelType parallelType() const override { return mParallelType; }

    /**
     * @brief Set the parallel type of this algorithm.
     * 
     * Use gwmodel_set_GWAverage_openmp() to set parallel type of this algorithm to ParallelType::OpenMP in shared build.
     * 
     * @param type Parallel type of this algorithm.
     */
    void setParallelType(const ParallelType& type) override;

public:     // IParallelOpenmpEnabled

    /**
     * @brief Set the thread numbers while paralleling.
     * 
     * Use gwmodel_set_GWAverage_openmp() to set this property in shared build.
     * 
     * @param threadNum Number of threads.
     */
    void setOmpThreadNum(const int threadNum) override { mOmpThreadNum = threadNum; }

    /**
     * @brief \~english Update calculator function according to parallel type and mode.
     */
    void updateCalculator();

private:
    /**
     * @brief \~english GWAverage algorithm implemented with no parallel methods. \~chinese GWAverage算法的单线程实现。
     */
    void GWAverageSerial();

    /**
     * @brief \~english GWCorrelation algorithm implemented with no parallel methods. \~chinese GWCorrelation算法的单线程实现。
     */
    void GWCorrelationSerial();

#ifdef ENABLE_OPENMP
    /**
     * @brief \~english GWAverage algorithm implemented with OpenMP. \~chinese GWAverage算法的多线程实现。
     */
    void GWAverageOmp();

    /**
     * @brief \~english GWCorrelation algorithm implemented with OpenMP. \~chinese GWCorrelation算法的多线程实现。
     */
    void GWCorrelationOmp();
#endif

    void calibration(const arma::mat& locations, const arma::mat& x);

private:
    bool mQuantile = false;             //!< \~english Indicator of whether calculate quantile statistics. \~chinese 是否使用基于排序的算法
    bool mIsCorrWithFirstOnly = false;  //!< \~english Indicator of whether calculate local correlations and covariances between the first variable and the other variables. \~chinese 是否仅为第一个变量计算与其他变量的相关系数

    bool mIsAutoselectBandwidth = false;//!< \~english Whether need bandwidth autoselect. \~chinese 是否需要自动优选带宽。
    bool mIsAutoselectLambda = false;//!< \~english Whether need lambda autoselect. \~chinese 是否需要自动优选lambda。
    BandwidthSelectionCriterionType mBandwidthSelectionCriterion = BandwidthSelectionCriterionType::AIC;//!< \~english Bandwidth Selection Criterion Type. \~chinese 默认的带宽优选方式。
    BandwidthSelectionCriterionCalculator mBandwidthSelectionCriterionFunction = &GWAverage::bandwidthSizeCriterionCVSerial;//!< \~english Bandwidth Selection Criterion Function. \~chinese 默认的带宽优选函数。
    BandwidthCriterionList mBandwidthSelectionCriterionList;//!< \~english Bandwidth Selection Criterion List. \~chinese 默认的带宽优选参数列表。
    double mBandwidthLastCriterion = DBL_MAX;   //!< \~english Last criterion for bandwidth selection. \~chinese 上一次带宽优选的有效指标值。

    arma::mat mX;             //!< \~english Variable matrix \~chinese 变量矩阵
    arma::mat mLocalMean;     //!< \~english Local mean \~chinese 局部均值
    arma::mat mStandardDev;   //!< \~english Local standard deviation \~chinese 局部标准差
    arma::mat mLocalSkewness; //!< \~english Local skewness \~chinese 局部偏度
    arma::mat mLCV;           //!< \~english Local coefficients of variation \~chinese 局部变化系数
    arma::mat mLVar;          //!< \~english Local variance \~chinese 局部方差
    arma::mat mLocalMedian;   //!< \~english Local medians \~chinese 局部中位数
    arma::mat mIQR;           //!< \~english Local interquartile ranges \~chinese 局部分位距
    arma::mat mQI;            //!< \~english Local quantile imbalances and coordinates \~chinese 局部分位数不平衡度
    arma::mat mCovmat;        //!< \~english Local covariances \~chinese 局部协方差
    arma::mat mCorrmat;       //!< \~english Local correlations (Pearson's) \~chinese 局部皮尔逊相关系数
    arma::mat mSCorrmat;      //!< \~english Local correlations (Spearman's) \~chinese 局部斯皮尔曼相关系数

    GWAverageMode mGWAverageMode = GWAverageMode::Average;       //!< \~english GWAverage working mode \~chinese GWAverage的工作模式

    SummaryCalculator mSummaryFunction = &GWAverage::GWAverageSerial;  //!< \~english Calculator for summary statistics \~chinese 计算函数
    ParallelType mParallelType = ParallelType::SerialOnly;  //!< \~english Parallel type \~chinese 并行方法
    int mOmpThreadNum = 8;                                  //!< \~english Numbers of threads to be created while paralleling \~chinese 多线程所使用的线程数
};

}


#endif  // GWAVERAGE_H