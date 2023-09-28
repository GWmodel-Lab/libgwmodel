#ifndef ISAPATIALCUDAENABLED_H
#define ISAPATIALCUDAENABLED_H

#include <cuda_runtime.h>

/**
 * @brief \~english Interface for enable CUDA acceleration for spatial weights. \~chinese 空间权重计算启用CUDA加速的接口。
 * 
 */
struct ISpatialCudaEnabled
{
    /**
     * @brief \~english Get whether to use CUDA. \~chinese 获取是否使用CUDA。
     * 
     * @return true \~english Use CUDA \~chinese 使用CUDA
     * @return false \~english Do not use CUDA \~chinese 不使用CUDA
     */
    virtual bool useCuda() = 0;

    /**
     * @brief \~english Set whether to use CUDA. \~chinese 设置是否使用CUDA。
     * 
     * @param isUseCuda \~english Whether to use CUDA \~chinese 是否使用CUDA
     */
    virtual void setUseCuda(bool isUseCuda) = 0;

    /**
     * @brief \~english Prepare environment for CUDA computing. \~chinese 准备 CUDA 计算的环境。
     * 
     * @param gpuId \~english The ID of selected GPU \~chinese 选择的 GPU 的 ID。
     * @return cudaError_t \~english CUDA error success \~chinese CUDA 错误或者成功。
     */
    virtual cudaError_t prepareCuda(size_t gpuId) = 0;
};

#endif  // ISAPATIALCUDAENABLED_H
