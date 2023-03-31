/**
 * @file GWSS.h
 * @author HPDell (hu_yigong@whu.edu.cn)
 * @brief This file define GWSS, which is used for Geographically Weighted Summary Statistics. 
 * @version 0.1
 * @date 2020-10-11
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#ifndef GWSS_H
#define GWSS_H

#include "SpatialMonoscaleAlgorithm.h"
#include "IMultivariableAnalysis.h"
#include "IParallelizable.h"

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
 * - local mean <- GWSS::localMean()
 * - local standard deviation <- GWSS::localSDev()
 * - local variance <- GWSS::localVar()
 * - local skewness <- GWSS::localSkewness()
 * - local coefficients of variation <- GWSS::localCV()
 * - local covariances <- GWSS::localCov()
 * - local correlations (Pearson's) <- GWSS::localCorr()
 * - local correlations (Spearman's) <- GWSS::localSCorr()
 * - local medians <- GWSS::localMedian()
 * - local interquartile ranges <- GWSS::iqr()
 * - local quantile imbalances and coordinates <- GWSS::qi()
 * 
 * \~chinese
 * @brief 地理加权汇总统计分析算法类。
 * 地理加权汇总统计是计算局部加权统计值的方法。
 * 可计算的统计值包括： local mean, local standard deviation, local variance, local skewness, local coefficients of variation, 
 * local covariances, local correlations (Pearson's), local correlations (Spearman's),
 * local medians, local interquartile ranges, local quantile imbalances and coordinates.
 * 使用下面这些函数获取上述值：
 * 
 * - local mean <- GWSS::localMean()
 * - local standard deviation <- GWSS::localSDev()
 * - local variance <- GWSS::localVar()
 * - local skewness <- GWSS::localSkewness()
 * - local coefficients of variation <- GWSS::localCV()
 * - local covariances <- GWSS::localCov()
 * - local correlations (Pearson's) <- GWSS::localCorr()
 * - local correlations (Spearman's) <- GWSS::localSCorr()
 * - local medians <- GWSS::localMedian()
 * - local interquartile ranges <- GWSS::iqr()
 * - local quantile imbalances and coordinates <- GWSS::qi()
 */
class GWSS : public SpatialMonoscaleAlgorithm, public IMultivariableAnalysis, public IParallelizable, public IParallelOpenmpEnabled
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

    typedef void (GWSS::*SummaryCalculator)();  //!< \~english Calculator for summary statistics \~chinese 汇总统计计算函数

protected:
    static arma::vec findq(const arma::mat& x, const arma::vec& w);

public:
    
    /**
     * @brief \~english Construct a new GWSS object. \~chinese 构造一个新的 GWSS 对象。
     * 
     */
    GWSS() {}
    
    /**
     * @brief \~english Construct a new GWSS object. \~chinese 构造一个新的 GWSS 对象。
     * 
     */
    GWSS(const arma::mat x, const arma::mat coords, const SpatialWeight& spatialWeight)
        : SpatialMonoscaleAlgorithm(spatialWeight, coords)
    {
        mX = x;
    }

    /**
     * @brief \~english Destroy the GWSS object. \~chinese 销毁 GWSS 对象。
     * 
     */
    ~GWSS() {}

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
    arma::mat localMean() const { return mLocalMean; }
    
    /**
     * @brief \~english Get local standard deviation on each sample. \~chinese 获取每个样本的局部标准差。
     * 
     * @return \~english Local standard deviation on each sample \~chinese 每个样本的局部标准差
     */
    arma::mat localSDev() const { return mStandardDev; }
    
    /**
     * @brief \~english Get local skewness on each sample. \~chinese 获取每个样本的局部偏度。
     * 
     * @return \~english Local skewness on each sample \~chinese 每个样本的局部偏度
     */
    arma::mat localSkewness() const { return mLocalSkewness; }
    
    /**
     * @brief \~english Get local coefficients of variation on each sample. \~chinese 获取每个样本的局部变化系数。
     * 
     * @return \~english Local coefficients of variation on each sample \~chinese 每个样本的局部变化系数
     */
    arma::mat localCV() const { return mLCV; }
    
    /**
     * @brief \~english Get local variance on each sample. \~chinese 获取每个样本的局部方差。
     * 
     * @return \~english Local variance on each sample \~chinese 每个样本的局部方差
     */
    arma::mat localVar() const { return mLVar; }

    
    /**
     * @brief \~english Get local median on each sample. \~chinese 获取每个样本的局部中位数。
     * 
     * @return \~english Local median on each sample \~chinese 每个样本的局部中位数
     */
    arma::mat localMedian() const { return mLocalMedian; }
    
    /**
     * @brief \~english Get local interquartile ranges on each sample. \~chinese 获取每个样本的局部四分位距。
     * 
     * @return \~english Local interquartile ranges on each sample \~chinese 每个样本的局部四分位距
     */
    arma::mat iqr() const { return mIQR; }
    
    /**
     * @brief \~english Get local quantile imbalances and coordinates on each sample. \~chinese 获取每个样本的局部分位数不平衡度。
     * 
     * @return \~english Local quantile imbalances and coordinates on each sample \~chinese 每个样本的局部分位数不平衡度
     */
    arma::mat qi() const { return mQI; }

    
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
    arma::mat localCov() const { return mCovmat; }
    
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
    arma::mat localCorr() const { return mCorrmat; }
    
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
    arma::mat localSCorr() const { return mSCorrmat; }

public:     // SpatialAlgorithm interface
    bool isValid() override;

public:     // IMultivariableAnalysis
    arma::mat variables() const override { return mX; }
    
    /**
     * @brief \~english set variables \~chinese 设置变量x。
     * 
     * @param x \~english variables for gwss \~chinese 进行GWSS的变量，如果只有一列，只能进行GWAverage。
     */
    void setVariables(const arma::mat& x) override { mX = x; }

    // void setSummaryFunction(bool SummaryFunc)
    /**
     * @brief \~english set gwss Function, true for GWAverage, false for GWCorrelation 
     * \~chinese 设置GWSS的函数mSummaryFunction，ture使用GWAverage，false使用GWCorrelation。
     * 
     * @param SummaryFunc \~english true for GWAverage, false for GWCorrelation \~chinese ture使用GWAverage，false使用GWCorrelation。
     */
    void isGWSSAverage(bool SummaryFunc)
    {
        gwssFunc = SummaryFunc;
        mSummaryFunction=gwssFunc? &GWSS::GWAverageSerial : &GWSS::GWCorrelationSerial;
    }

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
     * Use gwmodel_set_gwss_openmp() to set parallel type of this algorithm to ParallelType::OpenMP in shared build.
     * 
     * @param type Parallel type of this algorithm.
     */
    void setParallelType(const ParallelType& type) override;

public:     // IParallelOpenmpEnabled

    /**
     * @brief Set the thread numbers while paralleling.
     * 
     * Use gwmodel_set_gwss_openmp() to set this property in shared build.
     * 
     * @param threadNum Number of threads.
     */
    void setOmpThreadNum(const int threadNum) override { mOmpThreadNum = threadNum; }

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

private:
    bool mQuantile = false;             //!< \~english Indicator of whether calculate quantile statistics. \~chinese 是否使用基于排序的算法
    bool mIsCorrWithFirstOnly = false;  //!< \~english Indicator of whether calculate local correlations and covariances between the first variable and the other variables. \~chinese 是否仅为第一个变量计算与其他变量的相关系数

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

    bool gwssFunc=true;       //!< \~english Which GWSS method to use, true for GWAverage, False for GWCorrelation \~chinese 使用哪一种GWSS方法，true代表GWAverage，False代表GWCorrelation

    SummaryCalculator mSummaryFunction = &GWSS::GWAverageSerial;  //!< \~english Calculator for summary statistics \~chinese 计算函数
    ParallelType mParallelType = ParallelType::SerialOnly;  //!< \~english Parallel type \~chinese 并行方法
    int mOmpThreadNum = 8;                                  //!< \~english Numbers of threads to be created while paralleling \~chinese 多线程所使用的线程数
};

}


#endif  // GWSS_H