#ifndef GWPCA_H
#define GWPCA_H

#include <armadillo>
#include <vector>
#include <tuple>
#include "SpatialMonoscaleAlgorithm.h"
#include "IMultivariableAnalysis.h"
#include "IParallelizable.h"


namespace gwm
{

/**
 * @brief \~english Geographically weighted principle component analysis. \~chinese 地理加权主成分分析。
 * 
 */
class GWPCA: public SpatialMonoscaleAlgorithm, public IMultivariableAnalysis
{
private:
    typedef arma::mat (GWPCA::*Solver)(const arma::mat&, arma::cube&, arma::mat&);  //!< \~english Calculator to solve \~chinese 模型求解函数

public: // Constructors and Deconstructors

    /**
     * @brief \~english Construct a new GWPCA object. \~chinese 构造一个新的 GWPCA 对象。
     * 
     */
    GWPCA() {}

    /**
     * @brief \~english Construct a new GWPCA object. \~chinese 构造一个新的 GWPCA 对象。
     * 
     * @param x \~english Variables \~chinese 变量
     * @param coords \~english Coordinates \~chinese 样本坐标
     * @param spatialWeight \~english Spatial weighting scheme \~chinese 空间权重配置
     */
    GWPCA(const arma::mat x, const arma::mat coords, const SpatialWeight& spatialWeight)
        : SpatialMonoscaleAlgorithm(spatialWeight, coords)
    {
        mX = x;
    }
    
    /**
     * @brief \~english Destroy the GWPCA object. \~chinese 销毁 GWPCA 对象。
     * 
     */
    virtual ~GWPCA() {}

    /**
     * @brief \~english Get the number of Kept Components. \~chinese 获取保留主成分数量。
     * 
     * @return int \~english Number of Kept Components \~chinese 保留主成分数量
     */
    int keepComponents() { return mK; }

    /**
     * @brief \~english Set the number of Kept Components object. \~chinese 设置保留主成分数量。
     * 
     * @param k \~english Number of Kept Components \~chinese 保留主成分数量
     */
    void setKeepComponents(int k) { mK = k; }

    /**
     * @brief \~english Get the Local Principle Values matrix. \~chinese 获取局部主成分值。
     * 
     * @return arma::mat \~english Local Principle Values matrix \~chinese 局部主成分值
     */
    arma::mat localPV() { return mLocalPV; }

    /**
     * @brief \~english Get the Loadings matrix. \~chinese 获取局部载荷矩阵。
     * 
     * @return arma::mat \~english Loadings matrix \~chinese 局部载荷矩阵
     */
    arma::cube loadings() { return mLoadings; }

    /**
     * @brief \~english Get the Standard deviation matrix. \~chinese 获取标准差矩阵。
     * 
     * @return arma::mat \~english Standard deviation matrix \~chinese 标准差矩阵
     */
    arma::mat sdev() { return mSDev; }

    /**
     * @brief \~english Get the Scores matrix. \~chinese 获取得分矩阵。
     * 
     * @return arma::mat \~english Scores matrix \~chinese 得分矩阵
     */
    arma::cube scores() { return mScores; }

public: // IMultivariableAnalysis
    virtual arma::mat variables() const override { return mX; }
    virtual void setVariables(const arma::mat& x) override { mX = x; }
    virtual void run() override;

public: // Algorithm
    virtual bool isValid() override;

private:

    /**
     * @brief \~english Function to carry out PCA. \~chinese 执行 PCA 的函数。
     * 
     * @param x \~english Symmetric data matrix \~chinese 对称数据矩阵
     * @param loadings [out] \~english Out reference to loadings matrix \~chinese 载荷矩阵
     * @param sdev [out] \~english Out reference to standard deviation matrix \~chinese 标准差
     * @return arma::mat \~english Principle values matrix \~chinese 主成分值矩阵
     */
    arma::mat pca(const arma::mat& x, arma::cube& loadings, arma::mat& sdev)
    {
        return (this->*mSolver)(x, loadings, sdev);
    }

    /**
     * @brief \~english Serial version of PCA funtion. \~chinese 单线程 PCA 函数。
     * 
     * @param x \~english Symmetric data matrix \~chinese 对称数据矩阵
     * @param loadings [out] \~english Out reference to loadings matrix \~chinese 载荷矩阵
     * @param sdev [out] \~english Out reference to standard deviation matrix \~chinese 标准差
     * @return arma::mat \~english Principle values matrix \~chinese 主成分值矩阵
     */
    arma::mat solveSerial(const arma::mat& x, arma::cube& loadings, arma::mat& sdev);

    /**
     * @brief \~english Function to carry out weighted PCA. \~chinese 执行加权PCA的函数。
     * 
     * @param x \~english Symmetric data matrix \~chinese 对称数据矩阵
     * @param w \~english Weight vector \~chinese 权重向量
     * @param V [out] \~english Right orthogonal matrix \~chinese 右边的正交矩阵
     * @param d [out] \~english Rectangular diagonal matri \~chinese 矩形对角阵
     */
    void wpca(const arma::mat& x, const arma::vec& w, arma::mat& V, arma::vec & d);

private:    // Algorithm Parameters
    int mK = 2;  //!< \~english Number of components to be kept \~chinese 要保留的主成分数量
    // bool mRobust = false;

private:    // Algorithm Results
    arma::mat mLocalPV;               //!< \~english Local principle component values \~chinese 局部主成分值
    arma::cube mLoadings;             //!< \~english Loadings for each component \~chinese 局部载荷矩阵
    arma::mat mSDev;                  //!< \~english Standard Deviation \~chinese 标准差矩阵
    arma::cube mScores;               //!< \~english Scores for each variable \~chinese 得分矩阵
    arma::uvec mWinner;               //!< \~english Winner variable at each sample \~chinese 优胜变量索引值

private:    // Algorithm Runtime Variables
    arma::mat mX;           //!< \~english Variable matrix \~chinese 变量矩阵
    arma::vec mLatestWt;    //!< \~english Latest weigths \~chinese 最新的权重

    Solver mSolver = &GWPCA::solveSerial;   //!< \~english Calculator to solve \~chinese 模型求解函数
};

}

#endif  // GWPCA_H