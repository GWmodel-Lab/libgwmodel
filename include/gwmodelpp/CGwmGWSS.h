/**
 * @file CGwmGWSS.h
 * @author HPDell (hu_yigong@whu.edu.cn)
 * @brief This file define CGwmGWSS, which is used for Geographically Weighted Summary Statistics. 
 * @version 0.1
 * @date 2020-10-11
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#ifndef CGWMGWSS_H
#define CGWMGWSS_H

#include "CGwmSpatialMonoscaleAlgorithm.h"
#include "IGwmMultivariableAnalysis.h"
#include "IGwmParallelizable.h"

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
 *     - CGwmGWSS::mSourceLayer
 *     - CGwmGWSS::mVariables
 *     - CGwmGWSS::mSpatialWeight
 * 3. Run the algorithm by calling CGwmGWSS::run()
 * 
 * When finished, the algorithm will store those local statistics on each sample in different matrices. 
 * To get these matrices, call these functions:
 * 
 * - local mean <- CGwmGWSS::localMean()
 * - local standard deviation <- CGwmGWSS::localSDev()
 * - local variance <- CGwmGWSS::localVar()
 * - local skewness <- CGwmGWSS::localSkewness()
 * - local coefficients of variation <- CGwmGWSS::localCV()
 * - local covariances <- CGwmGWSS::localCov()
 * - local correlations (Pearson's) <- CGwmGWSS::localCorr()
 * - local correlations (Spearman's) <- CGwmGWSS::localSCorr()
 * - local medians <- CGwmGWSS::localMedian()
 * - local interquartile ranges <- CGwmGWSS::iqr()
 * - local quantile imbalances and coordinates <- CGwmGWSS::qi()
 * 
 * All these matrices are also in CGwmGWSS::mResultLayer, which is usually returned by CGwmGWSS::resultLayer().
 * These matrices are arranged according to the order above, their name is stored in the member CGwmSimpleLayer::mFields. 
 * 
 * For more details on how to use this class, see /test/GWSS/static.cpp or /test/GWSS/shared.cpp .
 */
class CGwmGWSS : public CGwmSpatialMonoscaleAlgorithm, public IGwmMultivariableAnalysis, public IGwmOpenmpParallelizable
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
    static double covwt(const mat &x1, const mat &x2, const vec &w)
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
    static double corwt(const mat &x1, const mat &x2, const vec &w)
    {
        return covwt(x1,x2,w)/sqrt(covwt(x1,x1,w)*covwt(x2,x2,w));
    }

    static vec del(vec x,int rowcount);

    static vec rank(vec x)
    {
        vec n = linspace(0.0, (double)x.n_rows - 1, x.n_rows);
        vec res = n(sort_index(x));
        return n(sort_index(res)) + 1.0;
    }

    enum NameFormat
    {
        PrefixVarName,
        SuffixVarName,
        PrefixVarNamePair,
        SuffixVarNamePair
    };

    typedef tuple<string, mat, NameFormat> ResultLayerDataItem;

    typedef void (CGwmGWSS::*SummaryCalculator)();

protected:
    static vec findq(const mat& x, const vec& w);

public:
    
    /**
     * @brief Construct a new CGwmGWSS object.
     * 
     * Use gwmodel_create_gwss_algorithm() to construct an instance in shared build.
     */
    CGwmGWSS();

    /**
     * @brief Destroy the CGwmGWSS object.
     * 
     * Use gwmodel_create_gwss_algorithm() to destruct an instance in shared build.
     */
    ~CGwmGWSS();

    /**
     * @brief Get the CGwmGWSS::mQuantile object .
     * 
     * @return true if CGwmGWSS::mQuantile is true.
     * @return false if CGwmGWSS::mQuantile is false.
     */
    bool quantile() const;

    /**
     * @brief Set the CGwmGWSS::mQuantile object.
     * 
     * @param quantile The value for CGwmGWSS::mQuantile.
     */
    void setQuantile(bool quantile);

    /**
     * @brief Set the CGwmGWSS::mIsCorrWithFirstOnly object.
     * 
     * Use gwmodel_set_gwss_options() to set this property in shared build.
     * 
     * @return true if CGwmGWSS::mIsCorrWithFirstOnly is true.
     * @return false if CGwmGWSS::mIsCorrWithFirstOnly is false.
     */
    bool isCorrWithFirstOnly() const;

    /**
     * @brief Set the CGwmGWSS::mIsCorrWithFirstOnly object
     * 
     * @param corrWithFirstOnly The value for CGwmGWSS::mIsCorrWithFirstOnly.
     */
    void setIsCorrWithFirstOnly(bool corrWithFirstOnly);

    /**
     * @brief Get the CGwmGWSS::mLocalMean object. 
     * 
     * Use gwmodel_get_gwss_local_mean() to get this property in shared build.
     * 
     * @return Local mean on each sample.
     * The number of rows is the same as number of features. 
     * The number of columns is the same as number of fields, arranged in the same order as CGwmGWSS::mVariables.
     */
    mat localMean() const { return mLocalMean; }
    
    /**
     * @brief Get the CGwmGWSS::mStandardDev object. 
     * 
     * Use gwmodel_get_gwss_local_sdev() to get this property in shared build.
     * 
     * @return Local standard deviation on each sample.
     * The number of rows is the same as number of features. 
     * The number of columns is the same as number of fields, arranged in the same order as CGwmGWSS::mVariables.
     */
    mat localSDev() const { return mStandardDev; }
    
    /**
     * @brief Get the CGwmGWSS::mLocalSkewness object. 
     * 
     * Use gwmodel_get_gwss_local_skew() to get this property in shared build.
     * 
     * @return Local skewness on each sample.
     * The number of rows is the same as number of features. 
     * The number of columns is the same as number of fields, arranged in the same order as CGwmGWSS::mVariables.
     */
    mat localSkewness() const { return mLocalSkewness; }
    
    /**
     * @brief Get the CGwmGWSS::mLCV object. 
     * 
     * Use gwmodel_get_gwss_local_cv() to get this property in shared build.
     * 
     * @return Local coefficients of variation on each sample.
     * The number of rows is the same as number of features. 
     * The number of columns is the same as number of fields, arranged in the same order as CGwmGWSS::mVariables.
     */
    mat localCV() const { return mLCV; }
    
    /**
     * @brief Get the CGwmGWSS::mLVar object. 
     * 
     * Use gwmodel_get_gwss_local_var() to get this property in shared build.
     * 
     * @return Local variance on each sample.
     * The number of rows is the same as number of features. 
     * The number of columns is the same as number of fields, arranged in the same order as CGwmGWSS::mVariables.
     */
    mat localVar() const { return mLVar; }

    
    /**
     * @brief Get the CGwmGWSS::mLocalMedian object. 
     * 
     * Use gwmodel_get_gwss_local_median() to get this property in shared build.
     * 
     * @return Local median on each sample.
     * The number of rows is the same as number of features. 
     * The number of columns is the same as number of fields, arranged in the same order as CGwmGWSS::mVariables.
     */
    mat localMedian() const { return mLocalMedian; }
    
    /**
     * @brief Get the CGwmGWSS::mIQR object. 
     * 
     * Use gwmodel_get_gwss_local_iqr() to get this property in shared build.
     * 
     * @return Local interquartile ranges on each sample.
     * The number of rows is the same as number of features. 
     * The number of columns is the same as number of fields, arranged in the same order as CGwmGWSS::mVariables.
     */
    mat iqr() const { return mIQR; }
    
    /**
     * @brief Get the CGwmGWSS::mQI object. 
     * 
     * Use gwmodel_get_gwss_local_qi() to get this property in shared build.
     * 
     * @return Local quantile imbalances and coordinates on each sample.
     * The number of rows is the same as number of features. 
     * The number of columns is the same as number of fields, arranged in the same order as CGwmGWSS::mVariables.
     */
    mat qi() const { return mQI; }

    
    /**
     * @brief Get the CGwmGWSS::mCovmat object. 
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
    mat localCov() const { return mCovmat; }
    
    /**
     * @brief Get the CGwmGWSS::mCorrmat object. 
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
    mat localCorr() const { return mCorrmat; }
    
    /**
     * @brief Get the CGwmGWSS::mSCorrmat object. 
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
    mat localSCorr() const { return mSCorrmat; }

public:     // GwmAlgorithm interface;
    /**
     * @brief Run the algorithm.
     * 
     * Use gwmodel_run_gwss() to run this algorithm in shared build.
     * 
     */
    void run() override;

public:     // GwmSpatialAlgorithm interface
    bool isValid() override;

public:     // IGwmMultivariableAnalysis
    vector<GwmVariable> variables() const override;
    void setVariables(const vector<GwmVariable>& variables) override;

public:     // IGwmParallelizable
    int parallelAbility() const override;
    ParallelType parallelType() const override;

    /**
     * @brief Set the parallel type of this algorithm.
     * 
     * Use gwmodel_set_gwss_openmp() to set parallel type of this algorithm to ParallelType::OpenMP in shared build.
     * 
     * @param type Parallel type of this algorithm.
     */
    void setParallelType(const ParallelType& type) override;

public:     // IGwmOpenmpParallelizable

    /**
     * @brief Set the thread numbers while paralleling.
     * 
     * Use gwmodel_set_gwss_openmp() to set this property in shared build.
     * 
     * @param threadNum Number of threads.
     */
    void setOmpThreadNum(const int threadNum) override;

private:

    /**
     * @brief Set CGwmGWSS::mX according to layer and variables. 
     * If there are \f$n\f$ features in layer and \f$k\f$ elements in variables, this function will set matrix x to the shape \f$ n \times k \f$.
     * Its element in location \f$ (i,j) \f$ will equal to the value of i-th feature's j-th variable. 
     * 
     * @param x Reference of CGwmGWSS::mX or other matrix to store value of variables.
     * @param layer Pointer to source data layer. 
     * @param variables Vector of variables.
     */
    void setXY(mat& x, const CGwmSimpleLayer* layer, const vector<GwmVariable>& variables);

    /**
     * @brief Create a Distance Parameter object. Store in CGwmGWSS::mDistanceParameter.
     */
    void createDistanceParameter();

    /**
     * @brief Summary algorithm implemented with no parallel methods.
     */
    void summarySerial();

    /**
     * @brief Summary algorithm implemented with OpenMP.
     */
    void summaryOmp();

    /**
     * @brief Create a Result Layer object.
     * 
     * @param items Data used for creating result layer.
     */
    void createResultLayer(vector<ResultLayerDataItem> items);

private:
    vector<GwmVariable> mVariables;     //!< Variables specified for summary.

    bool mQuantile = false;             //!< Indicator of whether calculate quantile statistics.
    bool mIsCorrWithFirstOnly = false;  //!< Indicator of whether calculate local correlations and covariances between the first variable and the other variables.

    mat mX;             //!< Variable matrix.
    mat mLocalMean;     //!< Local mean.
    mat mStandardDev;   //!< Local standard deviation.
    mat mLocalSkewness; //!< Local skewness.
    mat mLCV;           //!< Local coefficients of variation.
    mat mLVar;          //!< Local variance.
    mat mLocalMedian;   //!< Local medians.
    mat mIQR;           //!< Local interquartile ranges.
    mat mQI;            //!< Local quantile imbalances and coordinates.
    mat mCovmat;        //!< Local covariances.
    mat mCorrmat;       //!< Local correlations (Pearson's).
    mat mSCorrmat;      //!< Local correlations (Spearman's).
    
    DistanceParameter* mDistanceParameter = nullptr;    //!< Distance parameter used in calling for CGwmSpatialWeight::weightVector() and CGwmMinkwoskiDistance::distance().

    SummaryCalculator mSummaryFunction = &CGwmGWSS::summarySerial;  //!< Summary function specified by CGwmGWSS::mParallelType.
    
    ParallelType mParallelType = ParallelType::SerialOnly;  //!< Parallel type.
    int mOmpThreadNum = 8;                                  //!< Numbers of threads to be created while paralleling.
};

inline bool CGwmGWSS::quantile() const
{
    return mQuantile;
}

inline void CGwmGWSS::setQuantile(bool quantile)
{
    mQuantile = quantile;
}

inline bool CGwmGWSS::isCorrWithFirstOnly() const
{
    return mIsCorrWithFirstOnly;
}

inline void CGwmGWSS::setIsCorrWithFirstOnly(bool corrWithFirstOnly)
{
    mIsCorrWithFirstOnly = corrWithFirstOnly;
}

inline vector<GwmVariable> CGwmGWSS::variables() const
{
    return mVariables;
}

inline void CGwmGWSS::setVariables(const vector<GwmVariable>& variables)
{
    mVariables = variables;
}

inline int CGwmGWSS::parallelAbility() const
{
    return ParallelType::SerialOnly
#ifdef ENABLE_OPENMP
        | ParallelType::OpenMP
#endif        
        ;
}

inline ParallelType CGwmGWSS::parallelType() const
{
    return mParallelType;
}

inline void CGwmGWSS::setOmpThreadNum(const int threadNum)
{
    mOmpThreadNum = threadNum;
}

#endif  // CGWMGWSS_H