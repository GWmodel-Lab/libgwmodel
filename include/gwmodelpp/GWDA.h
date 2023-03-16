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
     * They are local mean,  local variance, local skewness, local coefficients of variation,
     * local covariances, local correlations (Pearson's), local correlations (Spearman's),
     * local medians, local interquartile ranges, local quantile imbalances and coordinates.
     * To get these matrices, call these functions:
     *
     * - local mean <- GWDA::localMean()
     * - local variance <- GWDA::localVar()
     * - local skewness <- GWDA::localSkewness()
     * - local coefficients of variation <- GWDA::localCV()
     * - local covariances <- GWDA::localCov()
     *
     * \~chinese
     * @brief 地理加权分析算法类。
     * 地理加权汇总统计是计算局部加权统计值的方法。
     * 可计算的统计值包括： local mean, local standard deviation, local variance, local skewness, local coefficients of variation,
     * local covariances, local correlations (Pearson's), local correlations (Spearman's),
     * local medians, local interquartile ranges, local quantile imbalances and coordinates.
     * 使用下面这些函数获取上述值：
     *
     * - local mean <- GWDA::localMean()
     */
    //template<class T>
    class GWDA : public SpatialMonoscaleAlgorithm, public IMultivariableAnalysis, public IParallelizable, public IParallelOpenmpEnabled
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
         * @brief \~english Calculate weighted covariances for two matrices. \~chinese 计算两个矩阵的加权协方差mat。
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

        static arma::vec del(arma::vec x, arma::uword rowcount);

        static arma::vec rank(arma::vec x)
        {
            arma::vec n = arma::linspace(0.0, (double)x.n_rows - 1, x.n_rows);
            arma::vec res = n(sort_index(x));
            return n(sort_index(res)) + 1.0;
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
         *
         */
        bool isWqda() const { return mIsWqda; }

        /**
         *
         */
        void setIsWqda(bool iswqda)
        {
            mIsWqda = iswqda;
        }

        /**
         *
         */
        bool hasCov() const { return mHascov; }

        /**
         *
         */
        void setHascov(bool hascov)
        {
            mHascov = hascov;
        }

        /**
         *
         */
        bool hasMean() const { return mHasmean; }

        /**
         *
         */
        void setHasmean(bool hasmean)
        {
            mHasmean = hasmean;
        }

        /**
         *
         */
        bool hasPrior() const { return mHasprior; }

        /**
         *
         */
        void setHasprior(bool hasprior)
        {
            mHasprior = hasprior;
        }

        /**
         *
         */
        double correctRate() const { return mCorrectRate; }

        /**
         *
         */
        arma::mat res() const { return mRes; }

        /**
         *
         */
        std::vector<std::string> group() const { return mGroup; }

        /**
         *
         */
        arma::mat probs() const { return mProbs; }

        /**
         *
         */
        arma::mat pmax() const { return mPmax; }

        /**
         *
         */
        arma::mat entropy() const { return mEntropy; }


        /**
         *
         *
         */
        arma::mat wqda(arma::mat &x, std::vector<std::string> &y, arma::mat &wt, arma::mat &xpr, bool hasCOv, bool hasMean, bool hasPrior);

        arma::mat wlda(arma::mat &x, std::vector<std::string> &y, arma::mat &wt, arma::mat &xpr, bool hasCOv, bool hasMean, bool hasPrior);

        std::vector<arma::mat> splitX(arma::mat &x, std::vector<std::string> &y);

        arma::mat wMean(arma::mat &x, arma::mat &wt);

        arma::cube wVarCov(arma::mat &x, arma::mat &wt);

        arma::vec wPrior(arma::mat &wt, double sumW);

        arma::mat confusionMatrix(arma::mat &origin, arma::mat &classified);

        std::vector<std::string> levels(std::vector<std::string> &y);

        double shannonEntropy(arma::vec &p);

        arma::uvec findSameString(std::vector<std::string> &y,std::string s);

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
         * @brief \~english Summary algorithm implemented with no parallel methods. \~chinese 统计算法的单线程实现。
         */
        void discriminantAnalysisSerial();

#ifdef ENABLE_OPENMP
        /**
         * @brief \~english Summary algorithm implemented with OpenMP. \~chinese 统计算法的多线程实现。
         */
        void discriminantAnalysisOmp();
#endif

    private:
        bool mQuantile = false;            //!< \~english Indicator of whether calculate quantile statistics. \~chinese 是否使用基于排序的算法
        bool mIsCorrWithFirstOnly = false; //!< \~english Indicator of whether calculate local correlations and covariances between the first variable and the other variables. \~chinese 是否仅为第一个变量计算与其他变量的相关系数
        bool mIsWqda = false;
        bool mHascov = true;
        bool mHasmean = true;
        bool mHasprior = true;

        double mCorrectRate = 0;
    
        
        arma::mat mX; //!< \~english Variable matrix for training \~chinese 变量矩阵
        std::vector<std::string> mY; //!< \~english Variable vector \~chinese 变量矩阵
        arma::mat mprX; //!< \~english Variable matrix \~chinese 变量矩阵
        std::vector<std::string> mprY; //!< \~english Variable matrix \~chinese 变量矩阵
        arma::mat mRes; //!< \~english Variable matrix \~chinese 变量矩阵
        std::vector<std::string> mGroup; //!< \~english Variable matrix \~chinese 变量矩阵
        arma::mat mProbs; //!< \~english Variable matrix \~chinese 变量矩阵
        arma::mat mPmax; //!< \~english Variable matrix \~chinese 变量矩阵
        arma::mat mEntropy; //!< \~english Variable matrix \~chinese 变量矩阵

        DiscriminantAnalysisCalculator mDiscriminantAnalysisFunction = &GWDA::discriminantAnalysisSerial; //!< \~english Calculator for summary statistics \~chinese 汇总统计计算函数

        ParallelType mParallelType = ParallelType::SerialOnly; //!< \~english Parallel type \~chinese 并行方法
        int mOmpThreadNum = 8;                                 //!< \~english Numbers of threads to be created while paralleling \~chinese 多线程所使用的线程数
    };

}

#endif // GWDA_H