#ifndef SPATIALALGORITHM_H
#define SPATIALALGORITHM_H

#include "Algorithm.h"
#include <armadillo>

namespace gwm
{

/**
 * \~english
 * @brief Abstract spatial algorithm class. 
 * This class cannot been constructed.
 * It defines some interface commonly used in spatial algorithms.
 * 
 * \~chinese
 * @brief 空间算法的抽象基类。
 * 该类型不能被构造。
 * 定义了一些空间算法的常用接口。
 * 
 */
class SpatialAlgorithm : public Algorithm
{
public:

    /**
     * \~english
     * @brief Construct a new SpatialAlgorithm object.
     * 
     * \~chinese
     * @brief 构造一个新的 SpatialAlgorithm 对象。
     * 
     */
    SpatialAlgorithm() {}

    /**
     * \~english
     * @brief Construct a new SpatialAlgorithm object.
     * 
     * @param coords A coordinate matrix representing positions of samples.
     * 
     * \~chinese
     * @brief 构造一个新的 SpatialAlgorithm 对象。
     * 
     * @param coords 坐标矩阵，用于表示样本的位置。
     * 
     */
    SpatialAlgorithm(const arma::mat& coords) : mCoords(coords) {};

    /**
     * \~english
     * @brief Destroy the SpatialAlgorithm object.
     * 
     * \~chinese
     * @brief 销毁一个 SpatialAlgorithm 对象。
     * 
     */
    virtual ~SpatialAlgorithm() { mCoords.clear(); }

public:

    /**
     * \~english
     * @brief Get Coords.
     * 
     * @return arma::mat A coordinate matrix representing positions of samples.
     * 
     * \~chinese
     * @brief 获取坐标矩阵。
     * 
     * @return arma::mat 表示样本位置的坐标矩阵。
     * 
     */
    arma::mat coords()
    {
        return mCoords;
    }

    /**
     * \~english
     * @brief Set the Coords object
     * 
     * @param coords Coordinates representing positions of samples.
     * 
     * \~chinese
     * @brief 设置坐标矩阵。
     * 
     * @param coords 表示样本位置的坐标矩阵。
     * 
     */
    void setCoords(const arma::mat& coords)
    {
        mCoords = coords;
    }

    /**
     * \~english
     * @brief Check whether the algorithm's configuration is valid. 
     * 
     * @return true if the algorithm's configuration is valid.
     * @return false if the algorithm's configuration is invalid.
     * 
     * \~chinese
     * @brief 检查算法配置是否合法。
     * 
     * @return true 如果算法配置是合法的。
     * @return false 如果算法配置不合法。
     * 
     */
    virtual bool isValid() override;

protected:
    
    arma::mat mCoords;  //!< \~english Coordinate matrix  \~chinese 坐标矩阵
};

}

#endif  // SPATIALALGORITHM_H