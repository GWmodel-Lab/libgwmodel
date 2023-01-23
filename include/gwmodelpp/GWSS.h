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
 * @brief The class for Geographically Weighted Summary Statistics. 
 * Geographically Weighted Summary Statistics is an algorithm for calculating local weighted statistics. 
 * They are local mean, local standard deviation, local variance, local skewness, local coefficients of variation, 
 * local covariances, local correlations (Pearson's), local correlations (Spearman's),
 * local medians, local interquartile ranges, local quantile imbalances and coordinates. 
 * 
 * To use this class, users need to follow this steps:
 * 
 * 1. Create an instance
 * 2. Set some properties, which are 
 *     - GWSS::mSourceLayer
 *     - GWSS::mVariables
 *     - GWSS::mSpatialWeight
 * 3. Run the algorithm by calling GWSS::run()
 * 
 * When finished, the algorithm will store those local statistics on each sample in different matrices. 
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
 * All these matrices are also in GWSS::mResultLayer, which is usually returned by GWSS::resultLayer().
 * These matrices are arranged according to the order above, their name is stored in the member SimpleLayer::mFields. 
 * 
 * For more details on how to use this class, see /test/GWSS/static.cpp or /test/GWSS/shared.cpp .
 */
class GWSS : public SpatialMonoscaleAlgorithm, public IMultivariableAnalysis, public IParallelizable, public IParallelOpenmpEnabled
{
public:

    /**
     * @brief Calculate weighted covariances for two matrices. 
     * 
     * @param x1 Matrix \f$ X_1 \f$.
     * @param x2 Matrix \f$ X_2 \f$.
     * @param w Weight vector \f$ w \f$.
     * @return weighted covariances 
     * \f[ cov(X_1,X_2) = \frac{\sum_{i=1}^n w_i(x_{1i} - \bar{x}_1) \sum_{i=1}^n w_i(x_{2i} - \bar{x}_2)}{1 - \sum_{i=1}^n w_i} \f]
     */
    static double covwt(const arma::mat &x1, const arma::mat &x2, const arma::vec &w)
    {
        return sum((sqrt(w) % (x1 - sum(x1 % w))) % (sqrt(w) % (x2 - sum(x2 % w)))) / (1 - sum(w % w));
    }

    /**
     * @brief Calculate weighted correlation for two matrices.
     * 
     * @param x1 Matrix \f$ X_1 \f$.
     * @param x2 Matrix \f$ X_2 \f$.
     * @param w Weight vector \f$ w \f$.
     * @return weighted correlation 
     * \f[ corr(X_1,X_2) = \frac{cov(X_1,X_2)}{\sqrt{cov(X_1,X_1) cov(X_2,X_2)}} \f]
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

    typedef void (GWSS::*SummaryCalculator)();

protected:
    static arma::vec findq(const arma::mat& x, const arma::vec& w);

public:
    
    /**
     * @brief Construct a new GWSS object.
     * 
     * Use gwmodel_create_gwss_algorithm() to construct an instance in shared build.
     */
    GWSS() {}
    
    /**
     * @brief Construct a new GWSS object.
     * 
     * Use gwmodel_create_gwss_algorithm() to construct an instance in shared build.
     */
    GWSS(const arma::mat x, const arma::mat coords, const SpatialWeight& spatialWeight)
        : SpatialMonoscaleAlgorithm(spatialWeight, coords)
    {
        mX = x;
    }

    /**
     * @brief Destroy the GWSS object.
     * 
     * Use gwmodel_create_gwss_algorithm() to destruct an instance in shared build.
     */
    ~GWSS() {}

    /**
     * @brief Get the GWSS::mQuantile object .
     * 
     * @return true if GWSS::mQuantile is true.
     * @return false if GWSS::mQuantile is false.
     */
    bool quantile() const { return mQuantile; }

    /**
     * @brief Set the GWSS::mQuantile object.
     * 
     * @param quantile The value for GWSS::mQuantile.
     */
    void setQuantile(bool quantile) { mQuantile = quantile; }

    /**
     * @brief Set the GWSS::mIsCorrWithFirstOnly object.
     * 
     * Use gwmodel_set_gwss_options() to set this property in shared build.
     * 
     * @return true if GWSS::mIsCorrWithFirstOnly is true.
     * @return false if GWSS::mIsCorrWithFirstOnly is false.
     */
    bool isCorrWithFirstOnly() const { return mIsCorrWithFirstOnly; }

    /**
     * @brief Set the GWSS::mIsCorrWithFirstOnly object
     * 
     * @param corrWithFirstOnly The value for GWSS::mIsCorrWithFirstOnly.
     */
    void setIsCorrWithFirstOnly(bool corrWithFirstOnly) { mIsCorrWithFirstOnly = corrWithFirstOnly; }

    /**
     * @brief Get the GWSS::mLocalMean object. 
     * 
     * Use gwmodel_get_gwss_local_mean() to get this property in shared build.
     * 
     * @return Local mean on each sample.
     * The number of rows is the same as number of features. 
     * The number of columns is the same as number of fields, arranged in the same order as GWSS::mVariables.
     */
    arma::mat localMean() const { return mLocalMean; }
    
    /**
     * @brief Get the GWSS::mStandardDev object. 
     * 
     * Use gwmodel_get_gwss_local_sdev() to get this property in shared build.
     * 
     * @return Local standard deviation on each sample.
     * The number of rows is the same as number of features. 
     * The number of columns is the same as number of fields, arranged in the same order as GWSS::mVariables.
     */
    arma::mat localSDev() const { return mStandardDev; }
    
    /**
     * @brief Get the GWSS::mLocalSkewness object. 
     * 
     * Use gwmodel_get_gwss_local_skew() to get this property in shared build.
     * 
     * @return Local skewness on each sample.
     * The number of rows is the same as number of features. 
     * The number of columns is the same as number of fields, arranged in the same order as GWSS::mVariables.
     */
    arma::mat localSkewness() const { return mLocalSkewness; }
    
    /**
     * @brief Get the GWSS::mLCV object. 
     * 
     * Use gwmodel_get_gwss_local_cv() to get this property in shared build.
     * 
     * @return Local coefficients of variation on each sample.
     * The number of rows is the same as number of features. 
     * The number of columns is the same as number of fields, arranged in the same order as GWSS::mVariables.
     */
    arma::mat localCV() const { return mLCV; }
    
    /**
     * @brief Get the GWSS::mLVar object. 
     * 
     * Use gwmodel_get_gwss_local_var() to get this property in shared build.
     * 
     * @return Local variance on each sample.
     * The number of rows is the same as number of features. 
     * The number of columns is the same as number of fields, arranged in the same order as GWSS::mVariables.
     */
    arma::mat localVar() const { return mLVar; }

    
    /**
     * @brief Get the GWSS::mLocalMedian object. 
     * 
     * Use gwmodel_get_gwss_local_median() to get this property in shared build.
     * 
     * @return Local median on each sample.
     * The number of rows is the same as number of features. 
     * The number of columns is the same as number of fields, arranged in the same order as GWSS::mVariables.
     */
    arma::mat localMedian() const { return mLocalMedian; }
    
    /**
     * @brief Get the GWSS::mIQR object. 
     * 
     * Use gwmodel_get_gwss_local_iqr() to get this property in shared build.
     * 
     * @return Local interquartile ranges on each sample.
     * The number of rows is the same as number of features. 
     * The number of columns is the same as number of fields, arranged in the same order as GWSS::mVariables.
     */
    arma::mat iqr() const { return mIQR; }
    
    /**
     * @brief Get the GWSS::mQI object. 
     * 
     * Use gwmodel_get_gwss_local_qi() to get this property in shared build.
     * 
     * @return Local quantile imbalances and coordinates on each sample.
     * The number of rows is the same as number of features. 
     * The number of columns is the same as number of fields, arranged in the same order as GWSS::mVariables.
     */
    arma::mat qi() const { return mQI; }

    
    /**
     * @brief Get the GWSS::mCovmat object. 
     * 
     * Use gwmodel_get_gwss_local_cov() to get this property in shared build.
     * 
     * @return Local coefficients of variation on each sample.
     * The number of rows is the same as number of features. 
     * If corrWithFirstOnly is set true, the number of columns is the (number of fields) - 1;
     * if not, the number of columns is the (((number of fields) - 1) * (number of fields)) / 2.
     * For variables \f$v_1, v_2, v_3, ... , v_{k-1}, v_k\f$, the fields are arranged as: 
     * \f$cov(v_1,v_2), cov(v_1,v_3), ... , cov(v_1,v_k), cov(v_2,v_3), ... , cov(v_2,v_k), ... , cov(v_{k-1},vk)\f$
     */
    arma::mat localCov() const { return mCovmat; }
    
    /**
     * @brief Get the GWSS::mCorrmat object. 
     * 
     * Use gwmodel_get_gwss_local_corr() to get this property in shared build.
     * 
     * @return Local correlations (Pearson's) on each sample.
     * The number of rows is the same as number of features. 
     * If corrWithFirstOnly is set true, the number of columns is the (number of fields) - 1;
     * if not, the number of columns is the (((number of fields) - 1) * (number of fields)) / 2.
     * For variables \f$v_1, v_2, v_3, ... , v_{k-1}, v_k\f$, the fields are arranged as: 
     * \f$corr(v_1,v_2), corr(v_1,v_3), ... , corr(v_1,v_k), corr(v_2,v_3), ... , corr(v_2,v_k), ... , corr(v_{k-1},vk)\f$
     */
    arma::mat localCorr() const { return mCorrmat; }
    
    /**
     * @brief Get the GWSS::mSCorrmat object. 
     * 
     * Use gwmodel_get_gwss_local_spearman_rho() to get this property in shared build.
     * 
     * @return Local correlations (Spearman's) on each sample.
     * The number of rows is the same as number of features. 
     * If corrWithFirstOnly is set true, the number of columns is the (number of fields) - 1;
     * if not, the number of columns is the (((number of fields) - 1) * (number of fields)) / 2.
     * For variables \f$v_1, v_2, v_3, ... , v_{k-1}, v_k\f$, the fields are arranged as: 
     * \f$corr(v_1,v_2), corr(v_1,v_3), ... , corr(v_1,v_k), corr(v_2,v_3), ... , corr(v_2,v_k), ... , corr(v_{k-1},vk)\f$
     */
    arma::mat localSCorr() const { return mSCorrmat; }

public:     // SpatialAlgorithm interface
    bool isValid() override;

public:     // IMultivariableAnalysis
    arma::mat variables() const override { return mX; }
    void setVariables(const arma::mat& x) override { mX = x; }
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
     * @brief Summary algorithm implemented with no parallel methods.
     */
    void summarySerial();

#ifdef ENABLE_OPENMP
    /**
     * @brief Summary algorithm implemented with OpenMP.
     */
    void summaryOmp();
#endif

private:
    bool mQuantile = false;             //!< Indicator of whether calculate quantile statistics.
    bool mIsCorrWithFirstOnly = false;  //!< Indicator of whether calculate local correlations and covariances between the first variable and the other variables.

    arma::mat mX;             //!< Variable matrix.
    arma::mat mLocalMean;     //!< Local mean.
    arma::mat mStandardDev;   //!< Local standard deviation.
    arma::mat mLocalSkewness; //!< Local skewness.
    arma::mat mLCV;           //!< Local coefficients of variation.
    arma::mat mLVar;          //!< Local variance.
    arma::mat mLocalMedian;   //!< Local medians.
    arma::mat mIQR;           //!< Local interquartile ranges.
    arma::mat mQI;            //!< Local quantile imbalances and coordinates.
    arma::mat mCovmat;        //!< Local covariances.
    arma::mat mCorrmat;       //!< Local correlations (Pearson's).
    arma::mat mSCorrmat;      //!< Local correlations (Spearman's).
    
    SummaryCalculator mSummaryFunction = &GWSS::summarySerial;  //!< Summary function specified by GWSS::mParallelType.
    
    ParallelType mParallelType = ParallelType::SerialOnly;  //!< Parallel type.
    int mOmpThreadNum = 8;                                  //!< Numbers of threads to be created while paralleling.
};

}


#endif  // GWSS_H