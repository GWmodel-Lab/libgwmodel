#ifndef IGWMPARALLELIZABLE_H
#define IGWMPARALLELIZABLE_H

enum ParallelType
{
    SerialOnly = 1 << 0,
    OpenMP = 1 << 1,
    CUDA = 1 << 2
};

struct IGwmParallelizable
{
    virtual int parallelAbility() const = 0;
    virtual ParallelType parallelType() const = 0;
    virtual void setParallelType(const ParallelType& type) = 0;
};

struct IGwmOpenmpParallelizable : public IGwmParallelizable
{
    virtual void setOmpThreadNum(const int threadNum) = 0;
};

struct IGwmCudaParallelizable : public IGwmParallelizable
{
    virtual void setGPUId(const int gpuId) = 0;
    virtual void setGroupSize(const double size) = 0;
};

#endif  // IGWMPARALLELIZABLE_H