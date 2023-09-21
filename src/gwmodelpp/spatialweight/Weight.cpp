#include "gwmodelpp/spatialweight/Weight.h"

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include "CudaUtils.h"
#endif

using namespace std;
using namespace gwm;

unordered_map<Weight::WeightType, string> Weight::TypeNameMapper = {
    std::make_pair(Weight::WeightType::BandwidthWeight, "BandwidthWeight")
};

#ifdef ENABLE_CUDA
cudaError_t Weight::prepareCuda(size_t gpuId)
{
    cudaDeviceProp devProp;
    checkCudaErrors(cudaGetDeviceProperties(&devProp, gpuId));
    mCudaThreads = devProp.maxThreadsPerBlock;
    mCudaPrepared = true;
    return cudaSuccess;
}
#endif // ENABLE_CUDA
