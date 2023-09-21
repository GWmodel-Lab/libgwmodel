#ifndef WEIGHT_H
#define WEIGHT_H

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include "gwmodelpp/spatialweight/cuda/ISpatialCudaEnabled.h"
#endif // ENABLE_CUDA

#include <unordered_map>
#include <string>
#include <armadillo>


namespace gwm
{

/**
 * @brief \~english Abstract base class for calculating weight from distance.
 * \~chinese 根据距离计算权重的基类。
 */
class Weight
#ifdef ENABLE_CUDA
    : public ISpatialCudaEnabled
#endif
{
public:

    /**
     * @brief \~english Type of weight. \~chinese 权重的类型。
     */
    enum WeightType
    {
        BandwidthWeight //!< \~english Bandwidth weight \~chinese 基于带宽的权重
    };

    static std::unordered_map<WeightType, std::string> TypeNameMapper;

public:

    /**
     * @brief \~english Construct a new Weight object. \~chinese 构造一个新的 Weight 对象。
     */
    Weight() {}

    /**
     * @brief \~english Destroy the Weight object. \~chinese 销毁 Weight 对象。
     */
    virtual ~Weight() {}

    /**
     * @brief \~english Clone this object. \~chinese 克隆该对象。
     * 
     * @return \~english Newly created pointer \~chinese 新创建的指针
     */
    virtual Weight* clone() = 0;

public:

    /**
     * @brief \~english Calculate weight vector from a distance vector.  \~chinese 从距离计算权重。
     * 
     * @param dist \~english According distance vector \~chinese 距离向量
     * @return \~english Weight vector \~chinese 权重向量
     */
    virtual arma::vec weight(arma::vec dist) = 0;

#ifdef ENABLE_CUDA
    bool useCuda() { return mUseCuda; }

    void setUseCuda(bool isUseCuda) { mUseCuda = isUseCuda; }

    virtual cudaError_t prepareCuda(size_t gpuId) override;
    
    virtual cudaError_t weight(double* d_dists, double* d_weights, size_t elems)
    {
        throw std::logic_error("Function not yet implemented");
    }
#endif

#ifdef ENABLE_CUDA
protected:
    bool mUseCuda = false;
    size_t mCudaThreads = 0;
    bool mCudaPrepared = false;
#endif // ENABLE_CUDA
};

}

#endif // WEIGHT_H
