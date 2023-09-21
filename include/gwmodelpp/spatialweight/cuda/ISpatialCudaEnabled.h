#ifndef ISAPATIALCUDAENABLED_H
#define ISAPATIALCUDAENABLED_H

#include <cuda_runtime.h>

struct ISpatialCudaEnabled
{
    virtual bool useCuda() = 0;
    virtual void setUseCuda(bool isUseCuda) = 0;
    virtual cudaError_t prepareCuda(size_t gpuId) = 0;
};

#endif  // ISAPATIALCUDAENABLED_H
