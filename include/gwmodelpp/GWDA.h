#ifndef GWDA_H
#define GWDA_H

#include "SpatialMonoscaleAlgorithm.h"
#include "IMultivariableAnalysis.h"
#include "IParallelizable.h"


namespace gwm
{

    /**
     * \~english
     * @brief The class for Geographically Weighted Discriminant Analysis.
     * Geographically Weighted Discriminant Analysis is an algorithm for calculating local weighted statistics,
     * where location-wise probabilities and their associated entropy are also calculated.
     *
     * \~chinese
     * @brief 地理加权判别分析类。
     * 地理加权判别分析是一种用于计算局部加权统计的算法，其中还计算位置概率及其相关熵。
     * 
     */
    //template<class T>
    class GWDA : public SpatialMonoscaleAlgorithm, public IMultivariableAnalysis, public IParallelizable, public IParallelOpenmpEnabled
    {
    public:

        /**
         * @brief \~english Calculate weighted covariances for two matrices. \~chinese 计算两个矩阵的加权协方差矩阵。
         *
         * @param x \~english Matrix \f$ X \f$ \~chinese 矩阵 \f$ X \f$
         * @param w \~english Weight vector \f$ w \f$ \~chinese 权重向量 \f$ w \f$
         * @return \~english Weighted covariances \f[ cov(X_1,X_2) = \frac{\sum_{i=1}^n w_i(x_{1i} - \bar{x}_1) \sum_{i=1}^n w_i(x_{2i} - \bar{x}_2)}{1 - \sum_{i=1}^n w_i} \f]
         * \~chinese 加权协方差 \f[ cov(X_1,X_2) = \frac{\sum_{i=1}^n w_i(x_{1i} - \bar{x}_1) \sum_{i=1}^n w_i(x_{2i} - \bar{x}_2)}{1 - \sum_{i=1}^n w_i} \f]
         */
        arma::mat covwtmat(const arma::mat &x, const arma::vec &wt);

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
            return covwt(x1, x2, w) / sqrt(covwt(x1, x1, w) * covwt(x2, x2, w));
        }

        typedef void (GWDA::*DiscriminantAnalysisCalculator)(); //!< \~english Calculator for summary statistics \~chinese 汇总统计计算函数

    public:
        /**
         * @brief \~english Construct a new GWDA object. \~chinese 构造一个新的 GWDA 对象。
         *
         */
        GWDA() {}

        /**
         * @brief \~english Construct a new GWDA object. \~chinese 构造一个新的 GWDA 对象。
         *
         */
        GWDA(const arma::mat x, const arma::mat coords, const SpatialWeight &spatialWeight)
            : SpatialMonoscaleAlgorithm(spatialWeight, coords)
        {
            mX = x;
        }

        /**
         * @brief \~english Destroy the GWDA object. \~chinese 销毁 GWDA 对象。
         *
         */
        ~GWDA() {}

        
        /**
         * @brief \~english Get whether weighted quadratic discriminant analysis will be applied; otherwise weighted linear discriminant analysis will be applied. \~chinese 获取是否应用加权二次判别分析；否则将应用加权线性判别分析。
         *
         * @return true \~english Yes \~chinese 是
         * @return false \~english No \~chinese 否
         */
        bool isWqda() const { return mIsWqda; }

        /**
         * @brief \~english Set whether weighted quadratic discriminant analysis will be applied. \~chinese 设置是否应用加权二次判别分析；否则将应用加权线性判别分析。
         *
         * @param isqwda \~english Whether weighted quadratic discriminant analysis will be applied. \~chinese 是否应用加权二次判别分析；否则将应用加权线性判别分析。
         */
        void setIsWqda(bool iswqda)
        {
            mIsWqda = iswqda;
        }

        /**
         * @brief \~english Get whether localised variance-covariance matrix is used for GW discriminant analysis; otherwise, global variance-covariance matrix is used. \~chinese 获取是否将局部方差协方差矩阵用于GW判别分析；否则，使用全局方差协方差矩阵。
         *
         * @return true \~english Yes \~chinese 是
         * @return false \~english No \~chinese 否
         */
        bool hasCov() const { return mHascov; }

        /**
         * @brief \~english Set whether localised variance-covariance matrix is used for GW discriminant analysis; otherwise, global variance-covariance matrix is used. \~chinese 设置将局部方差协方差矩阵用于GW判别分析；否则，使用全局方差协方差矩阵。
         *
         * @param hascov \~english Whether localised variance-covariance matrix is used for GW discriminant analysis; otherwise, global variance-covariance matrix is used. \~chinese 是否将局部方差协方差矩阵用于GW判别分析；否则，使用全局方差协方差矩阵。
         */
        void setHascov(bool hascov)
        {
            mHascov = hascov;
        }

        /**
         * @brief \~english Get whether localised mean is used for GW discriminant analysis; otherwise, global mean is used. \~chinese 获取是否使用局部平均值进行GW判别分析；否则，使用全局平均值。
         *
         * @return true \~english Yes \~chinese 是
         * @return false \~english No \~chinese 否
         */
        bool hasMean() const { return mHasmean; }

        /**
         * @brief \~english Set whether localised variance-covariance matrix is used for GW discriminant analysis; otherwise, global variance-covariance matrix is used. \~chinese 设置使用局部平均值进行GW判别分析；否则，使用全局平均值。
         *
         * @param hasmean \~english Whether localised mean is used for GW discriminant analysis; otherwise, global mean is used. \~chinese 是否使用局部平均值进行GW判别分析；否则，使用全局平均值。
         */
        void setHasmean(bool hasmean)
        {
            mHasmean = hasmean;
        }

        /**
         * @brief \~english Get whether localised prior probability is used for GW discriminant analysis; otherwise, fixed prior probability is used. \~chinese 获取是否将局部先验概率用于GW判别分析；否则，使用固定的先验概率。
         *
         * @return true \~english Yes \~chinese 是
         * @return false \~english No \~chinese 否
         */
        bool hasPrior() const { return mHasprior; }

        /**
         * @brief \~english Set whether localised prior probability is used for GW discriminant analysis; otherwise, fixed prior probability is used. \~chinese 设置将局部先验概率用于GW判别分析；否则，使用固定的先验概率。
         *
         * @param hasprior \~english Whether localised prior probability is used for GW discriminant analysis; otherwise, fixed prior probability is used. \~chinese 是否将局部先验概率用于GW判别分析；否则，使用固定的先验概率。
         */
        void setHasprior(bool hasprior)
        {
            mHasprior = hasprior;
        }

        /**
         * @brief \~english Get prediction accuracy \~chinese 获取预测正确率
         *
         * @return double \~english Prediction accuracy \~chinese 预测正确率
         */
        double correctRate() const { return mCorrectRate; }

        /**
         * @brief \~english Get the result matrix of geographical weighted discriminant analysis \~chinese 获取地理加权判别分析结果矩阵
         *
         * @return arma::mat \~english The result matrix of geographical weighted discriminant analysis \~chinese 地理加权判别分析结果矩阵
         */
        arma::mat res() const { return mRes; }

        /**
         * @brief \~english Get classification results \~chinese 获取分类结果
         *
         * @return std::vector<std::string> \~english Classification results \~chinese 分类结果
         */
        std::vector<std::string> group() const { return mGroup; }

        /**
         * @brief \~english Get location-wise probabilities \~chinese 获取位置概率
         *
         * @return arma::mat \~english Location-wise probabilities \~chinese 位置概率
         */
        arma::mat probs() const { return mProbs; }

        /**
         * @brief \~english Get max location-wise probabilities \~chinese 获取位置概率最大值
         *
         * @return arma::mat \~english max location-wise probabilities \~chinese 位置概率最大值
         */
        arma::mat pmax() const { return mPmax; }

        /**
         * @brief \~english Get associated entropy \~chinese 获取相关熵
         *
         * @return arma::mat \~english Associated entropy \~chinese 相关熵
         */
        arma::mat entropy() const { return mEntropy; }

        /**
         * \~english
         * @brief  Weighted quadratic discriminant analysis.
         * 
         * @param x Independent variables.
         * @param y Dependent variable.
         * @param wt Weighted matrix.
         * @param xpr Predict data variables.
         * @param hasCov whether localised variance-covariance matrix is used for GW discriminant analysis; otherwise, global variance-covariance matrix is used.
         * @param hasMean Whether localised mean is used for GW discriminant analysis; otherwise, global mean is used.
         * @param hasPrior Whether localised prior probability is used for GW discriminant analysis; otherwise, fixed prior probability is used.
         * 
         * @return arma::mat The result matrix of geographical weighted discriminant analysis.
         *
         * \~chinese
         * @brief  加权二次判别分析
         * 
         * @param x 自变量矩阵。
         * @param y 因变量。
         * @param wt 权重矩阵。
         * @param xpr 待预测数据变量
         * @param hasCov 是否使用局部平均值进行GW判别分析；否则，使用全局平均值。
         * @param hasMean 是否将局部方差协方差矩阵用于GW判别分析；否则，使用全局方差协方差矩阵。
         * @param hasPrior  是否将局部先验概率用于GW判别分析；否则，使用固定的先验概率。
         *
         * @return arma::mat 地理加权判别分析结果矩阵。
         */
        arma::mat wqda(arma::mat &x, std::vector<std::string> &y, arma::mat &wt, arma::mat &xpr, bool hasCOv, bool hasMean, bool hasPrior);

        /**
         * \~english
         * @brief  Weighted  linear discriminant analysis.
         * 
         * @param x Independent variables.
         * @param y Dependent variable.
         * @param wt Weighted matrix.
         * @param xpr Predict data variables.
         * @param hasCov whether localised variance-covariance matrix is used for GW discriminant analysis; otherwise, global variance-covariance matrix is used.
         * @param hasMean Whether localised mean is used for GW discriminant analysis; otherwise, global mean is used.
         * @param hasPrior Whether localised prior probability is used for GW discriminant analysis; otherwise, fixed prior probability is used.
         * 
         * @return arma::mat The result matrix of geographical weighted discriminant analysis.
         *
         * \~chinese
         * @brief  加权线性判别分析。
         * 
         * @param x 自变量矩阵。
         * @param y 因变量。
         * @param wt 权重矩阵。
         * @param xpr 待预测数据变量
         * @param hasCov 是否使用局部平均值进行GW判别分析；否则，使用全局平均值。
         * @param hasMean 是否将局部方差协方差矩阵用于GW判别分析；否则，使用全局方差协方差矩阵。
         * @param hasPrior  是否将局部先验概率用于GW判别分析；否则，使用固定的先验概率。
         *
         * @return arma::mat 地理加权判别分析结果矩阵。
         */
        arma::mat wlda(arma::mat &x, std::vector<std::string> &y, arma::mat &wt, arma::mat &xpr, bool hasCOv, bool hasMean, bool hasPrior);

        /**
         * @brief \~english Filter arguments according to different categories. \~chinese 根据不同类别筛选自变量矩阵。
         *
         * @param x \~english Independent variables. \~chinese 自变量矩阵。
         * @param y \~english Dependent variable. \~chinese 因变量。
         *
         * @return std::vector<arma::mat> \~english Filter results \~chinese 筛选结果。
         */
        std::vector<arma::mat> splitX(arma::mat &x, std::vector<std::string> &y);

        /**
         * @brief \~english Calculate the local weighted mean. \~chinese 计算局部加权平均值。
         *
         * @param x \~english Independent variables. \~chinese 自变量矩阵。
         * @param wt \~english Weighted matrix. \~chinese 权重矩阵。
         *
         * @return arma::mat \~english Calculate results \~chinese 计算结果。
         */
        arma::mat wMean(arma::mat &x, arma::mat &wt);

        /**
         * @brief \~english Calculate localised variance-covariance. \~chinese 计算局部方差协方差。
         *
         * @param x \~english Independent variables. \~chinese 自变量矩阵。
         * @param wt \~english Weighted matrix. \~chinese 权重矩阵。
         *
         * @return arma::mat \~english Calculate results \~chinese 计算结果。
         */
        arma::cube wVarCov(arma::mat &x, arma::mat &wt);

        /**
         * @brief \~english Calculate localised prior probability. \~chinese 计算局部先验概率。
         *
         * @param x \~english Independent variables. \~chinese 自变量矩阵。
         * @param sumW \~english Weight sum. \~chinese 权重和。
         *
         * @return arma::vec \~english Calculate results \~chinese 计算结果。
         */
        arma::vec wPrior(arma::mat &wt, double sumW);

        //arma::mat confusionMatrix(arma::mat &origin, arma::mat &classified);

        /**
         * @brief \~english Gets the category information of the dependent variable. \~chinese 获取因变量的类别信息。
         *
         * @param y \~english Dependent variable. \~chinese 因变量。
         *
         * @return std::vector<std::string> \~english Category information \~chinese 类别信息。
         */
        std::vector<std::string> levels(std::vector<std::string> &y);

        /**
         * @brief \~english Calculate associated entropy. \~chinese 计算关联熵。
         *
         * @param p \~english Location-wise probabilities \~chinese 位置概率
         *
         * @return arma::vec \~english Calculate results \~chinese 计算结果。
         */
        double shannonEntropy(arma::vec &p);

        /**
         * @brief \~english Gets the index of the same category in the dependent variable. \~chinese 获取因变量中相同类别的索引。
         *
         * @param y \~english Dependent variable. \~chinese 因变量。
         * @param s \~english Detection value. \~chinese 检测值。
         *
         * @return arma::uvec \~english Index information \~chinese 索引信息。
         */
        arma::uvec findSameString(std::vector<std::string> &y,std::string s);

        /**
         * @brief \~english Gets the counts of categories in the dependent variable. \~chinese 获取因变量中各个类别的数量。
         *
         * @param y \~english Dependent variable. \~chinese 因变量。
         *
         * @return arma::uvec \~english Counts information of each category. \~chinese 各个类别的数量信息。
         */
        std::unordered_map<std::string,arma::uword> ytable(std::vector<std::string> &y);

    public: // SpatialMonoscaleAlgorithm interface
        bool isValid() override;

    public: // IMultivariableAnalysis
        arma::mat variables() const override { return mX; }
        void setVariables(const arma::mat &x) override { mX = x; }
        void setGroup(std::vector<std::string> &y) { mY = y; }
        void run() override;

    public: // IParallelizable
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
        void setParallelType(const ParallelType &type) override;

    public: // IParallelOpenmpEnabled
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
         * @brief \~english GW discriminant analysis \~chinese 地理加权判别分析算法的单线程实现。
         */
        void discriminantAnalysisSerial();

#ifdef ENABLE_OPENMP
        /**
         * @brief \~english GW discriminant analysis \~chinese 地理加权判别分析算法的多线程实现。
         */
        void discriminantAnalysisOmp();
#endif

    private:
        bool mQuantile = false;            //!< \~english Indicator of whether calculate quantile statistics. \~chinese 是否使用基于排序的算法
        bool mIsCorrWithFirstOnly = false; //!< \~english Indicator of whether calculate local correlations and covariances between the first variable and the other variables. \~chinese 是否仅为第一个变量计算与其他变量的相关系数
        bool mIsWqda = false;        //!< \~english Whether weighted quadratic discriminant analysis will be applied. \~chinese 是否应用加权二次判别分析；否则将应用加权线性判别分析。
        bool mHascov = true;  //!< \~english Whether localised variance-covariance matrix is used for GW discriminant analysis; otherwise, global variance-covariance matrix is used. \~chinese 是否将局部方差协方差矩阵用于GW判别分析；否则，使用全局方差协方差矩阵。
        bool mHasmean = true; //!< \~english Whether localised mean is used for GW discriminant analysis; otherwise, global mean is used. \~chinese 是否使用局部平均值进行GW判别分析；否则，使用全局平均值。
        bool mHasprior = true; //!< \~english Whether localised prior probability is used for GW discriminant analysis; otherwise, fixed prior probability is used. \~chinese 是否将局部先验概率用于GW判别分析；否则，使用固定的先验概率。

        double mCorrectRate = 0; //!< \~english Prediction accuracy \~chinese 预测正确率
           
        arma::mat mX; //!< \~english Independent variable matrix for training \~chinese 自变量矩阵
        std::vector<std::string> mY; //!< \~english Dependent variable vector \~chinese 因变量矩阵
        arma::mat mprX; //!< \~english Variable Prediction independent variable matrix \~chinese 预测自变量矩阵
        std::vector<std::string> mprY; //!< \~english Prediction dependent variable matrix \~chinese 预测因变量矩阵
        arma::mat mRes; // !< \~english the result matrix of geographical weighted discriminant analysis \~chinese 地理加权判别分析结果矩阵
        std::vector<std::string> mGroup; //!< \~english Classification results \~chinese 分类结果
        arma::mat mProbs; //!< \~english Location-wise probabilities \~chinese 位置概率
        arma::mat mPmax; //!< \~english max location-wise probabilities \~chinese 位置概率最大值
        arma::mat mEntropy; //!< \~english Associated entropy \~chinese 相关熵

        DiscriminantAnalysisCalculator mDiscriminantAnalysisFunction = &GWDA::discriminantAnalysisSerial; //!< \~english GW discriminant analysis \~chinese 地理加权判别分析计算函数

        ParallelType mParallelType = ParallelType::SerialOnly; //!< \~english Parallel type \~chinese 并行方法
        int mOmpThreadNum = 8;                                 //!< \~english Numbers of threads to be created while paralleling \~chinese 多线程所使用的线程数
    };

}

#endif // GWDA_H