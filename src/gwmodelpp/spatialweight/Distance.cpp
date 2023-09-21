#include "gwmodelpp/spatialweight/Distance.h"
#include <assert.h>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include "CudaUtils.h"
#endif // ENABLE_CUDA

using namespace std;
using namespace gwm;

unordered_map<Distance::DistanceType, string> Distance::TypeNameMapper =
{
    std::make_pair(Distance::DistanceType::CRSDistance, "CRSDistance"),
    std::make_pair(Distance::DistanceType::MinkwoskiDistance, "MinkwoskiDistance"),
    std::make_pair(Distance::DistanceType::DMatDistance, "DMatDistance")
};

#ifdef ENABLE_CUDA
cudaError_t Distance::prepareCuda(size_t gpuId)
{
    cudaDeviceProp devProp;
    checkCudaErrors(cudaGetDeviceProperties(&devProp, gpuId));
    mCudaThreads = devProp.maxThreadsPerBlock;
    mCudaPrepared = true;
    return cudaSuccess;
}
#endif // ENABLE_CUDA
