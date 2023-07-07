#ifndef SPATIALMULTISCALEALGORITHM_H
#define SPATIALMULTISCALEALGORITHM_H

#include "SpatialAlgorithm.h"
#include <vector>
#include "spatialweight/SpatialWeight.h"

namespace gwm
{

/**
 * @brief Interface for spatial algorithm with a single bandwidth. 
 * It defines some interface commonly used in spatial algorithms:
 * 
 * - Getter and setter of spatial weight configuration.
 * 
 * Pointer of this type can be put to these classes:
 * 
 * - GWRBasic
 * - GWSS
 * 
 */
class SpatialMultiscaleAlgorithm : public SpatialAlgorithm
{
public:

    /**
     * @brief Construct a new SpatialMultiscaleAlgorithm object.
     */
    SpatialMultiscaleAlgorithm() {}

    /**
     * @brief Construct a new SpatialMultiscaleAlgorithm object
     * 
     * @param coords \~english Coordinates \~chinese 坐标
     * @param spatialWeights \~english Spatial weighting schemes \~chinese 空间加权配置
     */
    SpatialMultiscaleAlgorithm(const arma::mat& coords, const std::vector<SpatialWeight>& spatialWeights) : SpatialAlgorithm(coords)
    {
        mSpatialWeights = spatialWeights;
    }

    /**
     * @brief Destroy the SpatialMultiscaleAlgorithm object.
     */
    virtual ~SpatialMultiscaleAlgorithm() {}

public:

    /**
     * @brief Get the spatial weight configuration.
     * 
     * @return Spatial weight configuration object.
     */
    const std::vector<SpatialWeight>& spatialWeights() const;

    /**
     * @brief Set the spatial weight configuration.
     * 
     * Use gwmodel_set_gwr_spatial_weight() to set this property to GWRBasic in shared build.
     * 
     * Use gwmodel_set_gwss_spatial_weight() to set this property to GWSS in shared build.
     * 
     * @param spatialWeights Reference of spatial weight configuration object.
     */
    virtual void setSpatialWeights(const std::vector<SpatialWeight> &spatialWeights);
    
    /**
     * @brief \~english Create parameter for distance calculator. \~chinese 为距离计算函数创建参数
     * 
     * @param size \~english Number of parameters \~chinese 参数数量
     */
    void createDistanceParameter(size_t size);

protected:
    std::vector<SpatialWeight> mSpatialWeights;   //!< Spatial weight configuration.
};

inline const std::vector<SpatialWeight>& SpatialMultiscaleAlgorithm::spatialWeights() const
{
    return mSpatialWeights;
}

inline void SpatialMultiscaleAlgorithm::setSpatialWeights(const std::vector<SpatialWeight> &spatialWeights)
{
    mSpatialWeights = spatialWeights;
}

}

#endif  // SPATIALMULTISCALEALGORITHM_H