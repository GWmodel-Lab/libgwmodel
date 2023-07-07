#ifndef GWRGENERALIZED_H
#define GWRGENERALIZED_H

#include <utility>
#include <string>
#include <map>
#include <tuple>
#include <vector>
#include <initializer_list>
#include "GWRBase.h"
#include "RegressionDiagnostic.h"
#include "IBandwidthSelectable.h"
#include "IVarialbeSelectable.h"
#include "IParallelizable.h"
#include "GWRBasic.h"
#include "BandwidthSelector.h"
#include "BandwidthSelector.h"

namespace gwm
{

/**
 * @brief \~english Diagnostic information for generalized GWR model. \~chinese 广义地理加权模型的诊断信息。
 * 
 */
struct GWRGeneralizedDiagnostic
{
    double RSS;
    double AIC;
    double AICc;
    double RSquare;

    GWRGeneralizedDiagnostic()
    {
        AIC = 0.0;
        AICc = 0.0;
        RSS = 0.0;
        RSquare = 0.0;
    }

    GWRGeneralizedDiagnostic(const arma::vec &diag)
    {
        AIC = diag(0);
        AICc = diag(1);
        RSS = diag(2);
        RSquare = diag(3);
    }
};

/**
 * @brief \~english Diagnostic information for generalized linear model. \~chinese 广义线性模型的诊断信息。
 * 
 */
struct GLMDiagnostic
{
    double NullDev;
    double Dev;
    double AIC;
    double AICc;
    double RSquare;

    GLMDiagnostic()
    {
        AIC = 0.0;
        AICc = 0.0;
        Dev = 0.0;
        NullDev = 0.0;
        RSquare = 0.0;
    }

    GLMDiagnostic(const arma::vec &diag)
    {
        AIC = diag(0);
        AICc = diag(1);
        NullDev = diag(2);
        Dev = diag(3);
        RSquare = diag(4);
    }
};

/**
 * @brief \~english Generalized GWR. \~chinese 广义地理加权模型。
 * 
 */
class GWRGeneralized : public GWRBase, public IBandwidthSelectable, public IParallelizable, public IParallelOpenmpEnabled
{
public:

    /**
     * @brief \~english Family of generalized model. \~chinese 广义模型族。
     * 
     */
    enum Family
    {
        Poisson,    //!< \~english Poisson model \~chinese 泊松分布模型
        Binomial    //!< \~english Binomial model \~chinese 二项分布模型
    };

    /**
     * @brief \~english Get type of criterion for bandwidth selection. \~chinese 获取带宽自动优选指标值类型。
     * 
     * @return \~english BandwidthSelectionCriterionType Type of criterion for bandwidth selection. \~chinese 带宽自动优选指标值类型
     */
    enum BandwidthSelectionCriterionType
    {
        AIC,    //!< AIC
        CV      //!< CV
    };

    typedef double (GWRGeneralized::*BandwidthSelectCriterionFunction)(BandwidthWeight *);  //!< \~english Calculator to get criterion for bandwidth optimization \~chinese 带宽优选指标值计算函数
    typedef arma::mat (GWRGeneralized::*GGWRfitFunction)(const arma::mat& x, const arma::vec& y);   //!< \~english Calculator for fitting \~chinese 拟合函数
    typedef arma::vec (GWRGeneralized::*CalWtFunction)(const arma::mat &x, const arma::vec &y, arma::mat w);    //!< \~english Calculator for weighting \~chinese 加权函数

public:

    /**
     * @brief \~english Construct a new GWRGeneralized object \~chinese 构造一个新的 GWRGeneralized 对象 
     * 
     */
    GWRGeneralized(){};

    /**
     * @brief \~english Destroy the GWRGeneralized object \~chinese 销毁一个 GWRGeneralized 对象 
     * 
     */
    ~GWRGeneralized(){};

public: // IBandwidthSizeSelectable interface
    Status getCriterion(BandwidthWeight *bandwidthWeight, double& criterion) override
    {
        criterion = (this->*mBandwidthSelectCriterionFunction)(bandwidthWeight);
        return mStatus;
    }

public: // IRegressionAnalysis interface
    arma::mat predict(const arma::mat& locations) override;
    arma::mat fit() override;
    arma::mat fit(const arma::mat &x, const arma::vec &y, arma::mat &betasSE, arma::vec &shat, arma::vec &qdiag, arma::mat &S);

public: // IParallelalbe interface
    int parallelAbility() const override;

    ParallelType parallelType() const override;
    void setParallelType(const ParallelType &type) override;

public: // IParallelOpenmpEnabled interface
    void setOmpThreadNum(const int threadNum) override;

public:

    /**
     * @brief \~english Geographically weighted predicting. \~chinese 地理加权预测函数
     * 
     * @param x \~english Independent variables \~chinese 自变量
     * @param y \~english Dependent variables \~chinese 因变量
     * @param w \~english Weights \~chinese 权重
     * @return arma::vec \~english Coefficient estimates \~chinese 回归系数估计值
     */
    static arma::vec gwPredict(const arma::mat &x, const arma::vec &y, const arma::vec &w);

    /**
     * @brief \~english Geographically weighted fitting. \~chinese 地理加权拟合函数
     * 
     * @param x \~english Independent variables \~chinese 自变量
     * @param y \~english Dependent variables \~chinese 因变量
     * @param w \~english Weights \~chinese 权重
     * @param focus \~english Index of focused sample \~chinese 目标样本的索引值
     * @param ci \~english  \~chinese 
     * @param s_ri \~english  \~chinese 
     * @return arma::vec \~english Coefficient estimates \~chinese 回归系数估计值
     */
    static arma::vec gwFit(const arma::mat &x, const arma::vec &y, const arma::vec &w, arma::uword focus, arma::mat &ci, arma::mat &s_ri);

    static arma::mat dpois(arma::mat y, arma::mat mu);
    static arma::mat dbinom(arma::mat y, arma::mat m, arma::mat mu);
    static arma::mat lchoose(arma::mat n, arma::mat k);
    static arma::mat lgammafn(arma::mat x);

    static arma::mat CiMat(const arma::mat &x, const arma::vec &w);


protected:

    /**
     * @brief \~english Serial implementation of fitting Poisson model. \~chinese 泊松模型拟合的单线程实现。
     * 
     * @param x \~english Independent variables \~chinese 自变量
     * @param y \~english Dependent variables \~chinese 因变量
     * @return arma::vec \~english Coefficient estimates \~chinese 回归系数估计值
     */
    arma::mat fitPoissonSerial(const arma::mat& x, const arma::vec& y);

    /**
     * @brief \~english Serial implementation of fitting Binomial model. \~chinese 二项模型拟合的单线程实现。
     * 
     * @param x \~english Independent variables \~chinese 自变量
     * @param y \~english Dependent variables \~chinese 因变量
     * @return arma::vec \~english Coefficient estimates \~chinese 回归系数估计值
     */
    arma::mat fitBinomialSerial(const arma::mat& x, const arma::vec& y);

#ifdef ENABLE_OPENMP
    /**
     * @brief \~english Multithreading implementation of fitting Poisson model. \~chinese 泊松模型拟合的多线程实现。
     * 
     * @param x \~english Independent variables \~chinese 自变量
     * @param y \~english Dependent variables \~chinese 因变量
     * @return arma::vec \~english Coefficient estimates \~chinese 回归系数估计值
     */
    arma::mat fitPoissonOmp(const arma::mat& x, const arma::vec& y);

    /**
     * @brief \~english Multithreading implementation of fitting Binomial model. \~chinese 二项模型拟合的多线程实现。
     * 
     * @param x \~english Independent variables \~chinese 自变量
     * @param y \~english Dependent variables \~chinese 因变量
     * @return arma::vec \~english Coefficient estimates \~chinese 回归系数估计值
     */
    arma::mat fitBinomialOmp(const arma::mat& x, const arma::vec& y);
#endif

    arma::mat diag(arma::mat a);

    /**
     * @brief \~english Serial implementation of fitting weighted Poisson model. \~chinese 泊松加权模型拟合的单线程实现。
     * 
     * @param x \~english Independent variables \~chinese 自变量
     * @param y \~english Dependent variables \~chinese 因变量
     * @param w \~english Weights \~chinese 权重
     * @return arma::vec \~english Coefficient estimates \~chinese 回归系数估计值
     */
    arma::vec PoissonWtSerial(const arma::mat &x, const arma::vec &y, arma::mat w);

    /**
     * @brief \~english Serial implementation of fitting weighted Binomial model. \~chinese 二项加权模型拟合的单线程实现。
     * 
     * @param x \~english  \~chinese 
     * @param y \~english  \~chinese 
     * @param w \~english  \~chinese 
     * @return arma::vec \~english  \~chinese 
     */
    arma::vec BinomialWtSerial(const arma::mat &x, const arma::vec &y, arma::mat w);

#ifdef ENABLE_OPENMP
    /**
     * @brief \~english Serial implementation of fitting weighted Poisson model. \~chinese 泊松加权模型拟合的多线程实现。
     * 
     * @param x \~english  \~chinese 
     * @param y \~english  \~chinese 
     * @param w \~english  \~chinese 
     * @return arma::vec \~english  \~chinese 
     */
    arma::vec PoissonWtOmp(const arma::mat &x, const arma::vec &y, arma::mat w);

    /**
     * @brief \~english Serial implementation of fitting weighted Binomial model. \~chinese 二项加权模型拟合的多线程实现。
     * 
     * @param x \~english  \~chinese 
     * @param y \~english  \~chinese 
     * @param w \~english  \~chinese 
     * @return arma::vec \~english  \~chinese 
     */
    arma::vec BinomialWtOmp(const arma::mat &x, const arma::vec &y, arma::mat w);
#endif

    void CalGLMModel(const arma::mat& x, const arma::vec& y);

private:

    /**
     * @brief \~english Serial implementation of calculator to get CV criterion for given bandwidths. \~chinese 获取给定带宽值对应的CV值的串行实现。
     * 
     * @param bandwidths \~english Given bandwidths \~chinese 给定带宽值
     * @return double \~english Criterion value \~chinese 指标值
     */
    double bandwidthSizeGGWRCriterionCVSerial(BandwidthWeight *bandwidthWeight);

    /**
     * @brief \~english Serial implementation of calculator to get AIC criterion for given bandwidths. \~chinese 获取给定带宽值对应的AIC值的串行实现。
     * 
     * @param bandwidths \~english Given bandwidths \~chinese 给定带宽值
     * @return double \~english Criterion value \~chinese 指标值
     */
    double bandwidthSizeGGWRCriterionAICSerial(BandwidthWeight *bandwidthWeight);

#ifdef ENABLE_OPENMP
    /**
     * @brief \~english Multithreading implementation of calculator to get CV criterion for given bandwidths. \~chinese 获取给定带宽值对应的CV值的多线程实现。
     * 
     * @param bandwidths \~english Given bandwidths \~chinese 给定带宽值
     * @return double \~english Criterion value \~chinese 指标值
     */
    double bandwidthSizeGGWRCriterionCVOmp(BandwidthWeight *bandwidthWeight);

    /**
     * @brief \~english Multithreading implementation of calculator to get AIC criterion for given bandwidths. \~chinese 获取给定带宽值对应的AIC值的多线程实现。
     * 
     * @param bandwidths \~english Given bandwidths \~chinese 给定带宽值
     * @return double \~english Criterion value \~chinese 指标值
     */
    double bandwidthSizeGGWRCriterionAICOmp(BandwidthWeight *bandwidthWeight);
#endif

public:

    /**
     * @brief \~english Get the family of the model. \~chinese 获取模型的族。
     * 
     * @return Family \~english Family of the model \~chinese 模型的族
     */
    Family getFamily() const;

    /**
     * @brief \~english Set the family of the model. \~chinese 获取模型的族。
     * 
     * @param family \~english Family of the model \~chinese 模型的族
     */
    bool setFamily(Family family);

    /**
     * @brief \~english Get the tolerance. \~chinese 获取容忍值。
     * 
     * @return Family \~english Tolerance \~chinese 容忍值
     */
    double getTol() const;
    
    /**
     * @brief \~english Set the tolerance. \~chinese 获取容忍值。
     * 
     * @param tol \~english Tolerance \~chinese 容忍值
     */
    void setTol(double tol);

    /**
     * @brief \~english Get the maximum of iteration. \~chinese 获取最大迭代次数。
     * 
     * @return Family \~english Maximum of iteration \~chinese 最大迭代次数
     */
    size_t getMaxiter() const;

    /**
     * @brief \~english Set the maximum of iteration. \~chinese 获取最大迭代次数。
     * 
     * @param maxiter \~english Maximum of iteration \~chinese 最大迭代次数
     */
    void setMaxiter(std::size_t maxiter);

    arma::mat getWtMat1() const;

    arma::mat getWtMat2() const;

    /**
     * @brief \~english Get the diagnostic information. \~chinese 获取诊断信息。
     * 
     * @return GWRGeneralizedDiagnostic \~english Diagnostic information \~chinese 诊断信息
     */
    GWRGeneralizedDiagnostic getDiagnostic() const;

    /**
     * @brief \~english Get the diagnostic information of generalized linear model. \~chinese 获取广义线性模型的诊断信息。
     * 
     * @return GLMDiagnostic \~english Diagnostic information \~chinese 诊断信息
     */
    GLMDiagnostic getGLMDiagnostic() const;

    void setBandwidthSelectionCriterionType(const BandwidthSelectionCriterionType &bandwidthSelectionCriterionType);
    BandwidthCriterionList bandwidthSelectorCriterions() const;
    BandwidthCriterionList mBandwidthSelectionCriterionList;
    BandwidthSelectionCriterionType bandwidthSelectionCriterionType() const;

    /**
     * @brief \~english Get whether bandwidth optimization is enabled. \~chinese 获取是否进行带宽优选。
     * 
     * @return true \~english Yes \~chinese 是
     * @return false \~english No \~chinese 否
     */
    bool autoselectBandwidth() const;

    /**
     * @brief \~english Set whether bandwidth optimization is enabled. \~chinese 设置是否进行带宽优选。 
     * 
     * @param flag \~english Whether bandwidth optimization is enabled \~chinese 是否进行带宽优选
     */
    void setIsAutoselectBandwidth(bool value);

    arma::mat regressionData() const;
    void setRegressionData(const arma::mat &locations);

    /**
     * @brief \~english Get whether has hat matrix. \~chinese 获取是否有帽子矩阵。
     * 
     * @return true \~english Yes \~chinese 是
     * @return false \~english No \~chinese 否
     */
    bool hasHatMatrix() const;

    /**
     * @brief \~english Set whether has hat matrix. \~chinese 设置是否有帽子矩阵。 
     * 
     * @param flag \~english Whether has hat matrix. \~chinese 是否有帽子矩阵。 
     */
    void setHasHatMatrix(bool value);

    bool hasRegressionData() const;
    void setHasRegressionData(bool value);

private:

    /**
     * @brief \~english Create distance parameter for prediction. \~chinese 创建用于预测的距离度量参数。
     * 
     * @param locations \~english Locations to predict \~chinese 要预测的位置
     */
    void createPredictionDistanceParameter(const arma::mat& locations);

protected:
    Family mFamily;             //!< \~english Family of the model \~chinese 模型的族
    double mTol=1e-5;           //!< \~english Tolerance \~chinese 容忍值
    std::size_t mMaxiter=20;    //!< \~english Maximum of iteration \~chinese 最大迭代次数

    bool mHasHatMatrix = true;          //!< \~english Whether has hat matrix. \~chinese 是否有帽子矩阵。 
    bool mHasRegressionData = false;

    arma::mat mBetasSE; //!< //!< \~english Standard error of coefficient estimates \~chinese 回归系数估计值标准误差。
    arma::vec mShat;    //!< //!< \~english A vector of trace of \f$S\f$ and \f$S'S\f$ \~chinese 一个包含 \f$S\f$ 和 \f$S'S\f$ 矩阵迹的向量
    arma::mat mS;       //!< //!< \~english Hat matrix \f$S\f$ \~chinese 帽子矩阵 \f$S\f$
    double mGwDev;

    arma::mat mRegressionData;

    arma::mat mWtMat1;
    arma::mat mWtMat2;

    GWRGeneralizedDiagnostic mDiagnostic;   //!< \~english Diagnostic information \~chinese 诊断信息
    GLMDiagnostic mGLMDiagnostic;           //!< \~english Diagnostic information for GLM \~chinese GLM模型的诊断信息

    arma::mat mWt2;
    arma::mat myAdj;

    double mLLik = 0;   //!< \~english Logorithm of likelihood \~chinese 对数似然函数值

    GGWRfitFunction mGGWRfitFunction = &GWRGeneralized::fitPoissonSerial;   //!< \~english Calculator for fitting \~chinese 拟合函数
    CalWtFunction mCalWtFunction = &GWRGeneralized::PoissonWtSerial;        //!< \~english Calculator for weighting \~chinese 加权函数

    bool mIsAutoselectBandwidth = false;    //!< \~english Whether bandwidth optimization is enabled \~chinese 是否进行带宽优选
    BandwidthSelectionCriterionType mBandwidthSelectionCriterionType = BandwidthSelectionCriterionType::AIC;    //!< \~english Type of criterion for bandwidth optimization \~chinese 带宽优选指标类型
    BandwidthSelectCriterionFunction mBandwidthSelectCriterionFunction = &GWRGeneralized::bandwidthSizeGGWRCriterionCVSerial;   //!< \~english Calculator to get criterion for given bandwidth value \~chinese 用于根据给定带宽值计算指标值的函数
    BandwidthSelector mBandwidthSizeSelector;   //!< \~english Bandwidth size selector \~chinese 带宽选择器
    double mBandwidthLastCriterion = DBL_MAX;   //!< \~english Last criterion for bandwidth selection. \~chinese 上一次带宽优选的有效指标值。

    ParallelType mParallelType = ParallelType::SerialOnly;  //!< \~english Type of parallelization \~chinese 并行方法类型
    int mOmpThreadNum = 8;  //!< \~english Number of threads used in multithreading \~chinese 多线程所使用的线程数
};

inline GWRGeneralized::Family GWRGeneralized::getFamily() const
{
    return mFamily;
}

inline double GWRGeneralized::getTol() const
{
    return mTol;
}

inline size_t GWRGeneralized::getMaxiter() const
{
    return mMaxiter;
}

inline arma::mat GWRGeneralized::getWtMat1() const
{
    return mWtMat1;
}

inline arma::mat GWRGeneralized::getWtMat2() const
{
    return mWtMat2;
}

inline GWRGeneralizedDiagnostic GWRGeneralized::getDiagnostic() const
{
    return mDiagnostic;
}

inline GLMDiagnostic GWRGeneralized::getGLMDiagnostic() const
{
    return mGLMDiagnostic;
}

inline void GWRGeneralized::setTol(double tol)
{
    mTol = tol;
}

inline void GWRGeneralized::setMaxiter(size_t maxiter)
{
    mMaxiter = maxiter;
}

inline BandwidthCriterionList GWRGeneralized::bandwidthSelectorCriterions() const
{
    return mBandwidthSizeSelector.bandwidthCriterion();
}

inline bool GWRGeneralized::hasHatMatrix() const
{
    return mHasHatMatrix;
}

inline void GWRGeneralized::setHasHatMatrix(bool value)
{
    mHasHatMatrix = value;
}

inline bool GWRGeneralized::hasRegressionData() const
{
    return mHasRegressionData;
}

inline void GWRGeneralized::setHasRegressionData(bool value)
{
    mRegressionData = value;
}
inline arma::mat GWRGeneralized::regressionData() const
{
    return mRegressionData;
}

inline void GWRGeneralized::setRegressionData(const arma::mat &locations)
{
    mRegressionData = locations;
}

inline GWRGeneralized::BandwidthSelectionCriterionType GWRGeneralized::bandwidthSelectionCriterionType() const
{
    return mBandwidthSelectionCriterionType;
}

inline bool GWRGeneralized::autoselectBandwidth() const
{
    return mIsAutoselectBandwidth;
}

inline void GWRGeneralized::setIsAutoselectBandwidth(bool value)
{
    mIsAutoselectBandwidth = value;
}

inline int GWRGeneralized::parallelAbility() const
    {
        return ParallelType::SerialOnly
#ifdef ENABLE_OPENMP
            | ParallelType::OpenMP
#endif        
        ;
    }


inline ParallelType GWRGeneralized::parallelType() const
{
    return mParallelType;
}

inline void GWRGeneralized::setOmpThreadNum(const int threadNum)
{
    mOmpThreadNum = threadNum;
}

// inline arma::mat GWRGeneralized::fit(const arma::mat &x, const arma::vec &y, arma::mat &betasSE, arma::vec &shat, arma::vec &qdiag, arma::mat &S)
// {
//     switch (type)
//     {
//     case GWRGeneralized::Family::Poisson:
//         return regressionPoissonSerial(x, y);
//         break;
//     case GWRGeneralized::Family::Binomial:
//         return regressionBinomialSerial(x, y);
//         break;
//     default:
//         return regressionPoissonSerial(x, y);
//         break;
//     }
// }

}

#endif // GWRGENERALIZED_H