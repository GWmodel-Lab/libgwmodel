#ifndef SPATIALMONOSCALEALGORITHM_H
#define SPATIALMONOSCALEALGORITHM_H

#include "SpatialAlgorithm.h"
#include "spatialweight/SpatialWeight.h"

namespace gwm
{

/**
 * \~english
 * @brief Interface for spatial algorithm with a single bandwidth. 
 * It defines some interface commonly used in spatial monoscale algorithms.
 * 
 * \~chinese
 * @brief 空间单尺度算法基类。
 * 该类定义了一些空间但尺度算法的常用接口。
 * 
 */
class SpatialMonoscaleAlgorithm : public SpatialAlgorithm
{
public:

    /**
     * \~english
     * @brief Construct a new CGwmSpatialMonoscaleAlgorithm object.
     * 
     * \~chinese
     * @brief 构造一个 CGwmSpatialMonoscaleAlgorithm 类型。
     * 
     */
    SpatialMonoscaleAlgorithm() {}

    /**
     * \~english
     * @brief Construct a new CGwmSpatialMonoscaleAlgorithm object
     * 
     * @param spatialWeight Spatial weighting configuration.
     * @param coords Coordinates
     * 
     * \~chinese
     * @brief 构造一个 CGwmSpatialMonoscaleAlgorithm 类型。
     * 
     * @param spatialWeight 空间权重配置。
     * @param coords 坐标
     */
    SpatialMonoscaleAlgorithm(const SpatialWeight& spatialWeight, arma::mat coords) : SpatialAlgorithm(coords)
    {
        mSpatialWeight = spatialWeight;
    }

    /**
     * \~english
     * @brief Destroy the CGwmSpatialMonoscaleAlgorithm object.
     * 
     * \~chinese
     * @brief 析构 CGwmSpatialMonoscaleAlgorithm 对象。
     * 
     */
    virtual ~SpatialMonoscaleAlgorithm() {}

public:

    /**
     * \~english
     * @brief Get the spatial weight configuration.
     * 
     * @return Spatial weight configuration object.
     * 
     * \~chinese
     * @brief 获取空间权重配置。
     * 
     * @return 空间权重配置对象。
     * 
     */
    const SpatialWeight& spatialWeight() const
    {
        return mSpatialWeight;
    }

    /**
     * \~english
     * @brief Set the spatial weight configuration.
     * 
     * @param spatialWeight Reference of spatial weight configuration object.
     * 
     * \~chinese
     * @brief 设置空间权重配置对象。
     * 
     * @param spatialWeight 空间权重配置对象。
     * 
     */
    void setSpatialWeight(const SpatialWeight &spatialWeight)
    {
        mSpatialWeight = spatialWeight;
    }

    /**
     * \~english
     * @brief Create a Distance Parameter object. Store in CGwmGWSS::mDistanceParameter.
     * 
     * \~chinese
     * @brief 创建距离计算参数. 存储在 CGwmGWSS::mDistanceParameter 变量中。
     * 
     */
    void createDistanceParameter();

protected:
    SpatialWeight mSpatialWeight;   //!< Spatial weight configuration.    
};

}

#endif  // SPATIALMONOSCALEALGORITHM_H