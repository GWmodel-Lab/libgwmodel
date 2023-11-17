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
    virtual Weight* clone() const = 0;

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
    
    /**
     * @brief \~english Calculate weight vector from a distance vector.  \~chinese 从距离计算权重。
     * 
     * @param d_dists \~english Device pointer to distances \~chinese 指向输入距离的设备指针
     * @param d_weights \~english Device pointer to distances \~chinese 指向输出权重的设备指针
     * @param elems \~english Number of elements in distances \~chinese 距离向量的元素数量
     * @return cudaError_t \~english CUDA error or success \~chinese CUDA 错误或成功
     */
    virtual cudaError_t weight(double* d_dists, double* d_weights, size_t elems)
    {
        throw std::logic_error("Function not yet implemented");
    }
#endif

#ifdef ENABLE_CUDA
protected:
    bool mUseCuda = false;  //<! \~english Whether to use CUDA \~chinese 是否使用 CUDA
    int mGpuID = 0;  //<! \~english The ID of selected GPU \~chinese 选择的 GPU 的索引
    bool mCudaPrepared = false;  //<! \~english Whether CUDA has been prepared \~chinese CUDA 环境是否已经准备
    size_t mCudaThreads = 0;  //<! \~english Number of GPU threads \~chinese GPU 线程数
#endif // ENABLE_CUDA
};

}

#endif // WEIGHT_H
