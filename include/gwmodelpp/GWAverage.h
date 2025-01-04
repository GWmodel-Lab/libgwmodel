#ifndef GWAVERAGE_H
#define GWAVERAGE_H

#include "SpatialMonoscaleAlgorithm.h"
#include "IMultivariableAnalysis.h"
#include "IParallelizable.h"

namespace gwm
{

/**
 * \~english
 * @brief The class for Summary Statistics of Geographically Weighted Average. 
 * Geographically Weighted Average is an algorithm for calculating local weighted statistics. 
 * They are local mean, local standard deviation, local variance, local skewness, local coefficients of variation, 
 * local medians, local interquartile ranges, local quantile imbalances and coordinates.
 * To get these matrices, call these functions:
 * 
 * - local mean <- GWAverage::localMean()
 * - local standard deviation <- GWAverage::localSDev()
 * - local variance <- GWAverage::localVar()
 * - local skewness <- GWAverage::localSkewness()
 * - local coefficients of variation <- GWAverage::localCV()
 * - local medians <- GWAverage::localMedian()
 * - local interquartile ranges <- GWAverage::iqr()
 * - local quantile imbalances and coordinates <- GWAverage::qi()
 * 
 * \~chinese
 * @brief 地理加权汇总统计分析算法类。
 * 地理加权汇总统计是计算局部加权统计值的方法。
 * 可计算的统计值包括： local mean, local standard deviation, local variance, local skewness, local coefficients of variation, 
 * local medians, local interquartile ranges, local quantile imbalances and coordinates.
 * 使用下面这些函数获取上述值：
 * 
 * - local mean <- GWAverage::localMean()
 * - local standard deviation <- GWAverage::localSDev()
 * - local variance <- GWAverage::localVar()
 * - local skewness <- GWAverage::localSkewness()
 * - local coefficients of variation <- GWAverage::localCV()
 * - local medians <- GWAverage::localMedian()
 * - local interquartile ranges <- GWAverage::iqr()
 * - local quantile imbalances and coordinates <- GWAverage::qi()
 */
class GWAverage : public SpatialMonoscaleAlgorithm, public IMultivariableAnalysis, public IParallelizable, public IParallelOpenmpEnabled
{
public:

    static arma::vec del(arma::vec x, arma::uword rowcount);

    static arma::vec rank(arma::vec x)
    {
        arma::vec n = arma::linspace(0.0, (double)x.n_rows - 1, x.n_rows);
        arma::vec res = n(sort_index(x));
        return n(sort_index(res)) + 1.0;
    }

    typedef void (GWAverage::*SummaryCalculator)();  //!< \~english Calculator for summary statistics \~chinese 汇总统计计算函数

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

    void run() override;
    
    void calibration(const arma::mat& locations, const arma::mat& x);

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
     * @brief \~english Update calculator function according to parallel type.
     */
    void updateCalculator();

private:
    /**
     * @brief \~english GWAverage algorithm implemented with no parallel methods. \~chinese GWAverage算法的单线程实现。
     */
    void GWAverageSerial();

#ifdef ENABLE_OPENMP
    /**
     * @brief \~english GWAverage algorithm implemented with OpenMP. \~chinese GWAverage算法的多线程实现。
     */
    void GWAverageOmp();
#endif


private:
    bool mQuantile = false;             //!< \~english Indicator of whether calculate quantile statistics. \~chinese 是否使用基于排序的算法

    arma::mat mX;             //!< \~english Variable matrix \~chinese 变量矩阵
    arma::mat mLocalMean;     //!< \~english Local mean \~chinese 局部均值
    arma::mat mStandardDev;   //!< \~english Local standard deviation \~chinese 局部标准差
    arma::mat mLocalSkewness; //!< \~english Local skewness \~chinese 局部偏度
    arma::mat mLCV;           //!< \~english Local coefficients of variation \~chinese 局部变化系数
    arma::mat mLVar;          //!< \~english Local variance \~chinese 局部方差
    arma::mat mLocalMedian;   //!< \~english Local medians \~chinese 局部中位数
    arma::mat mIQR;           //!< \~english Local interquartile ranges \~chinese 局部分位距
    arma::mat mQI;            //!< \~english Local quantile imbalances and coordinates \~chinese 局部分位数不平衡度

    SummaryCalculator mSummaryFunction = &GWAverage::GWAverageSerial;  //!< \~english Calculator for summary statistics \~chinese 计算函数
    ParallelType mParallelType = ParallelType::SerialOnly;  //!< \~english Parallel type \~chinese 并行方法
    int mOmpThreadNum = 8;                                  //!< \~english Numbers of threads to be created while paralleling \~chinese 多线程所使用的线程数
};

}


#endif  // GWAVERAGE_H