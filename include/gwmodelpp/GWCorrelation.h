#ifndef GWCORRELATION_H
#define GWCORRELATION_H

#include "SpatialMultiscaleAlgorithm.h"
#include "IMultivariableAnalysis.h"
#include "IParallelizable.h"
#include "IBandwidthSelectable.h"

namespace gwm
{
#define GWM_LOG_TAG_GWCORR_INITIAL_BW "#initial-bandwidth "

/**
 * \~english
 * @brief The class for Geographically Weighted Correlation. 
 * Geographically Weighted Correlation is an algorithm for calculating local weighted statistics. 
 * They are local covariances, local correlations (Pearson's), local correlations (Spearman's),
 * local medians, local interquartile ranges, local quantile imbalances and coordinates.
 * To get these matrices, call these functions:
 * 
 * - local mean <- GWCorrelation::localMean()
 * - local standard deviation <- GWCorrelation::localSDev()
 * - local variance <- GWCorrelation::localVar()
 * - local skewness <- GWCorrelation::localSkewness()
 * - local coefficients of variation <- GWCorrelation::localCV()
 * - local covariances <- GWCorrelation::localCov()
 * - local correlations (Pearson's) <- GWCorrelation::localCorr()
 * - local correlations (Spearman's) <- GWCorrelation::localSCorr()
 * - local medians <- GWCorrelation::localMedian()
 * - local interquartile ranges <- GWCorrelation::iqr()
 * - local quantile imbalances and coordinates <- GWCorrelation::qi()
 * 
 * \~chinese
 * @brief 地理加权汇总统计分析算法类。
 * 地理加权汇总统计是计算局部加权统计值的方法。
 * 可计算的统计值包括： local covariances, local correlations (Pearson's), local correlations (Spearman's),
 * local medians, local interquartile ranges, local quantile imbalances and coordinates.
 * 使用下面这些函数获取上述值：
 * 
 * - local mean <- GWCorrelation::localMean()
 * - local standard deviation <- GWCorrelation::localSDev()
 * - local variance <- GWCorrelation::localVar()
 * - local skewness <- GWCorrelation::localSkewness()
 * - local coefficients of variation <- GWCorrelation::localCV()
 * - local covariances <- GWCorrelation::localCov()
 * - local correlations (Pearson's) <- GWCorrelation::localCorr()
 * - local correlations (Spearman's) <- GWCorrelation::localSCorr()
 * - local medians <- GWCorrelation::localMedian()
 * - local interquartile ranges <- GWCorrelation::iqr()
 * - local quantile imbalances and coordinates <- GWCorrelation::qi()
 */
class GWCorrelation : public SpatialMultiscaleAlgorithm, public IMultiresponseVariableAnalysis, public IParallelizable, public IParallelOpenmpEnabled, public IBandwidthSelectable
{
public:

    /**
     * \~english
     * @brief Type of bandwidth initilization.
     * 
     * \~chinese
     * @brief 带宽初始值类型。
     */
    enum BandwidthInitilizeType
    {
        Null,
        Initial,
        Specified
    };

    static std::unordered_map<BandwidthInitilizeType,std::string> BandwidthInitilizeTypeNameMapper; //!< \~english A mapper from bandwidth inilization types to their names. \~chinese 带宽初始值类型到其名称的映射表。

    /**
     * \~english
     * @brief Type of criterion for bandwidth selection.
     * 
     * \~chinese
     * @brief 带宽选择指标类型。
     */
    enum BandwidthSelectionCriterionType
    {
        CV,
        AIC
    };

    static std::unordered_map<BandwidthSelectionCriterionType,std::string> BandwidthSelectionCriterionTypeNameMapper;   //!< \~english A mapper from bandwidth selection criterion types to their names. \~chinese 带宽选择指标类型到其名称的映射表。


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

    /**
     * @brief \~english Construct a new GWCorrelation object. \~chinese 构造一个新的 GWCorrelation 对象。
     * 
     */
    GWCorrelation() {}

    /**
     * @brief \~english Destroy the GWCorrelation object. \~chinese 销毁 GWCorrelation 对象。
     * 
     */
    ~GWCorrelation() {}

    typedef void (GWCorrelation::*SummaryCalculator)();  //!< \~english Calculator for GWCorrelation \~chinese 计算函数

protected:
    static arma::vec findq(const arma::mat& x, const arma::vec& w);

public: //calculating functions

    // /**
    //  * @brief \~english Get whether calculate correlation between the first variable and others. \~chinese 获取是否仅为第一个变量计算与其他变量的相关系数
    //  * 
    //  * @return true \~english Yes \~chinese 是
    //  * @return false \~english No \~chinese 否
    //  */
    // bool isCorrWithFirstOnly() const { return mIsCorrWithFirstOnly; }

    // /**
    //  * @brief \~english Set whether calculate correlation between the first variable and others. \~chinese 设置是否仅为第一个变量计算与其他变量的相关系数 
    //  * 
    //  * @param corrWithFirstOnly \~english Whether calculate correlation between the first variable and others. \~chinese 是否仅为第一个变量计算与其他变量的相关系数
    //  */
    // void setIsCorrWithFirstOnly(bool corrWithFirstOnly) { mIsCorrWithFirstOnly = corrWithFirstOnly; }

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

public: //bandwidth selection

    typedef double (GWCorrelation::*BandwidthSizeCriterionFunction)(BandwidthWeight*);  //!< \~english Function to calculate the criterion for given bandwidth size. \~chinese 根据指定带宽大小计算对应指标值的函数。

    /**
     * \~english
     * @brief Declaration of criterion calculator for bandwidth selection. 
     * \~chinese
     * @brief 带宽优选指标计算函数声明。
     */
    typedef double (GWCorrelation::*BandwidthSelectionCriterionCalculator)(BandwidthWeight*);

    // bool isAutoselectBandwidth() const { return mIsAutoselectBandwidth; }
    // void setIsAutoselectBandwidth(bool isAutoSelect) { mIsAutoselectBandwidth = isAutoSelect; }

    /**
     * \~english
     * @brief Get criterion calculator function for optimize bandwidth size for one variable.
     * 
     * @param type The criterion type for optimize bandwidth size for one variable.
     * @return BandwidthSizeCriterionFunction The criterion calculator for optimize bandwidth size for one variable.
     * 
     * \~chinese
     * @brief 获取单个变量带宽优选的指标值计算函数。
     * 
     * @param type 单个变量带宽优选的指标值类型。
     * @return BandwidthSizeCriterionFunction 单个变量带宽优选的指标值计算函数。
     */
    BandwidthSizeCriterionFunction bandwidthSizeCriterionVar(BandwidthSelectionCriterionType type)
    {
#ifdef ENABLE_OPENMP
        if (mParallelType & ParallelType::OpenMP)
        {
            switch (type)
            {
            case BandwidthSelectionCriterionType::CV:
                return &GWCorrelation::bandwidthSizeCriterionCVOmp;
            case BandwidthSelectionCriterionType::AIC:
                return &GWCorrelation::bandwidthSizeCriterionAICOmp;
            default:
                return &GWCorrelation::bandwidthSizeCriterionAICOmp;
            }
        }
#endif // ENABLE_OPENMP
        switch (type)
        {
        case BandwidthSelectionCriterionType::CV:
            return &GWCorrelation::bandwidthSizeCriterionCVSerial;
        case BandwidthSelectionCriterionType::AIC:
            return &GWCorrelation::bandwidthSizeCriterionAICSerial;
        default:
            return &GWCorrelation::bandwidthSizeCriterionAICSerial;
        }
    }

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

#ifdef ENABLE_OPENMP
    double bandwidthSizeCriterionCVOmp(BandwidthWeight* bandwidthWeight);

    double bandwidthSizeCriterionAICOmp(BandwidthWeight* bandwidthWeight);
#endif

public:     // Implement IBandwidthSelectable
    Status getCriterion(BandwidthWeight* weight, double& criterion) override
    {
        criterion = (this->*mBandwidthSizeCriterion)(weight);
        return mStatus;
    }

    
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
     * @brief \~english GWCorrelation algorithm implemented with no parallel methods. \~chinese GWCorrelation算法的单线程实现。
     */
    void GWCorrelationSerial();

#ifdef ENABLE_OPENMP
    /**
     * @brief \~english GWCorrelation algorithm implemented with OpenMP. \~chinese GWCorrelation算法的多线程实现。
     */
    void GWCorrelationOmp();
#endif

public:     // IMultiresponseVariableAnalysis
    const arma::mat& independentVariables() const override { return mX; }
    
    /**
     * @brief \~english set variables \~chinese 设置变量x。
     * 
     * @param x \~english variables for GWAverage \~chinese 进行GWAverage的变量，如果只有一列，只能进行GWAverage。
     */
    void setIndependentVariables(const arma::mat& x) override { mX = x; }


    const arma::mat& responseVariables() const override { return mY; }
    
    /**
     * @brief \~english set variables \~chinese 设置变量x。
     * 
     * @param x \~english variables for GWAverage \~chinese 进行GWAverage的变量，如果只有一列，只能进行GWAverage。
     */
    void setResponseVariables(const arma::mat& y) override { mY = y; }

    void run() override;
    
    void calibration(const arma::mat& locations, const arma::mat& x);

protected:
    std::vector<SpatialWeight> mSpatialWeights;   //!< Spatial weight configuration.

    BandwidthWeight* bandwidth(size_t i)
    {
        return mSpatialWeights[i].weight<BandwidthWeight>();
    }

private:
    bool mQuantile = false;             //!< \~english Indicator of whether calculate quantile statistics. \~chinese 是否使用基于排序的算法
    // bool mIsCorrWithFirstOnly = false;  //是否仅为第一个变量计算与其他变量的相关系数

    // bool mIsAutoselectBandwidth = false;//!< \~english Whether need bandwidth autoselect. \~chinese 是否需要自动优选带宽。
    SpatialWeight mInitSpatialWeight;   //!< \~english Spatial weighting sheme for initializing bandwidth. \~chinese 计算初始带宽值时所用的空间权重配置。
    BandwidthSelectionCriterionType mBandwidthSelectionCriterion = BandwidthSelectionCriterionType::AIC;//!< \~english Bandwidth Selection Criterion Type. \~chinese 默认的带宽优选方式。
    BandwidthSizeCriterionFunction mBandwidthSizeCriterion = &GWCorrelation::bandwidthSizeCriterionCVSerial;//!< \~english Bandwidth Selection Criterion Function. \~chinese 默认的带宽优选函数。
    BandwidthCriterionList mBandwidthSelectionCriterionList;//!< \~english Bandwidth Selection Criterion List. \~chinese 默认的带宽优选参数列表。
    double mBandwidthLastCriterion = DBL_MAX;   //!< \~english Last criterion for bandwidth selection. \~chinese 上一次带宽优选的有效指标值。
    size_t mBandwidthSelectionCurrentIndex = 0; //!< \~english The index of variable which currently the algorithm select bandwidth for. \~chinese 当前正在选带宽的变量索引值。
    std::vector<BandwidthInitilizeType> mBandwidthInitilize;    //!< \~english Type of bandwidth initilization values. \~chinese 带宽初始值类型。
    std::vector<BandwidthSelectionCriterionType> mBandwidthSelectionApproach;   //!< \~english Type of bandwidth selection approach. \~chinese 带宽选择方法类型。

    arma::mat mX;             //!< \~english Variable matrix \~chinese 变量矩阵
    arma::mat mY;
    arma::mat mXi;
    arma::mat mYi;

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

    SummaryCalculator mSummaryFunction = &GWCorrelation::GWCorrelationSerial;  //!< \~english Calculator for summary statistics \~chinese 计算函数
    ParallelType mParallelType = ParallelType::SerialOnly;  //!< \~english Parallel type \~chinese 并行方法
    int mOmpThreadNum = 8;       

};

}

#endif  // GWCORRELATION_H