#include "gwmodelpp/spatialweight/BandwidthWeightKernel.h"

__global__ void gw_weight_gaussian_kernel(const double *d_dists, double bw, double *d_weights, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;
    int i = index;
    double dist = d_dists[i];
    d_weights[i] = exp((dist * dist) / ((-2)*(bw * bw)));
}


__global__ void gw_weight_exponential_kernel(const double *d_dists, double bw, double *d_weights, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;
    int i = index;
    double dist = d_dists[i];
    d_weights[i] = exp(-dist / bw);
}


__global__ void gw_weight_bisquare_kernel(const double *d_dists, double bw, double *d_weights, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;
    int i = index;
    double dist = d_dists[i];
    d_weights[i] = dist > bw ? 0 : (1 - (dist * dist) / (bw * bw))*(1 - (dist * dist) / (bw * bw));
}


__global__ void gw_weight_tricube_kernel(const double *d_dists, double bw, double *d_weights, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;
    int i = index;
    double dist = d_dists[i];
    d_weights[i] = dist > bw ? 0 :
        (1 - (dist * dist * dist) / (bw * bw * bw))*
        (1 - (dist * dist * dist) / (bw * bw * bw))*
        (1 - (dist * dist * dist) / (bw * bw * bw));
}


__global__ void gw_weight_boxcar_kernel(const double *d_dists, double bw, double *d_weights, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;
    int i = index;
    double dist = d_dists[i];
    d_weights[i] = dist > bw ? 0 : 1;
}


typedef void(*WEIGHT_KERNEL_CUDA)(const double*, double, double*, int);


const WEIGHT_KERNEL_CUDA GWRKernelCuda[5] = {
    gw_weight_gaussian_kernel,
    gw_weight_exponential_kernel,
    gw_weight_bisquare_kernel,
    gw_weight_tricube_kernel,
    gw_weight_boxcar_kernel
};


cudaError_t gw_weight_cuda(double bw, int kernel, bool adaptive, double *d_dists, double *d_weight, int ndp, int nrp, int threads)
{
    cudaError_t error;
    const WEIGHT_KERNEL_CUDA *kerf = GWRKernelCuda + kernel;
    switch (adaptive)
    {
        case true:
        {
            dim3 blockSize(threads), gridSize((ndp + blockSize.x - 1) / blockSize.x);
            for (size_t f = 0; f < nrp; f++)
            {
                // Backup d_dists, used for sort
                double *d_dists_bak;
                cudaMalloc((void **)&d_dists_bak, sizeof(double) * ndp);
                cudaMemcpy(d_dists_bak, d_dists + f * ndp, sizeof(double) * ndp, cudaMemcpyDeviceToDevice);
                thrust::device_ptr<double> v_dists(d_dists_bak);
                thrust::sort(v_dists, v_dists + ndp);
                // Calculate weight for each distance
                double bw_dist = v_dists[(int)(bw < ndp ? bw : ndp) - 1];
                (*kerf) << <gridSize, blockSize >> > (d_dists + f * ndp, bw_dist, d_weight + f * ndp, ndp);
                // Free d_dists_bak
                cudaFree(d_dists_bak);
                d_dists_bak = nullptr;
                // Get error
                error = cudaGetLastError();
                if (error != cudaSuccess)
                {
                        return error;
                }
            }
            break;
        }
        default:
        {
            dim3 blockSize(threads), gridSize((ndp * nrp + blockSize.x - 1) / blockSize.x);
            (*kerf) << <gridSize, blockSize >> > (d_dists, bw, d_weight, ndp * nrp);
            error = cudaGetLastError();
            if (error != cudaSuccess)
            {
                return error;
            }
            break;
        }
    }
    return cudaSuccess;
}


__global__ void gw_xtw_kernel(const double* d_x, const double* d_wights, int n, int k, double* d_xtw)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;
    int i = index;
    double weight = d_wights[index];
    for (int j = 0; j < k; j++)
    {
        int p = j + i * k;
        d_xtw[p] = d_x[p] * weight;
    }
}


cudaError_t gw_xtw_cuda(const double* d_x, const double* d_weight, int n, int k, double* d_xtw, int threads)
{
    cudaError_t error;
    dim3 blockSize(threads), gridSize((n + blockSize.x - 1) / blockSize.x);
    gw_xtw_kernel << <gridSize, blockSize >> > (d_x, d_weight, n, k, d_xtw);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        return error;
    }
    return cudaSuccess;
}


