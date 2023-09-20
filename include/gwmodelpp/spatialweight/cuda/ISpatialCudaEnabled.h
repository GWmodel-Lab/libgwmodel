#ifndef ISAPATIALCUDAENABLED_H
#define ISAPATIALCUDAENABLED_H

#include <cuda_runtime.h>

struct ISpatialCudaEnabled
{
    virtual bool useCuda() = 0;
    virtual void setUseCuda() = 0;
    virtual cudaError_t prepareCuda() = 0;
}

#endif  // ISAPATIALCUDAENABLED_H
