#include "gwmodelpp/spatialweight/cuda/CRSDistanceKernel.h"

#define M_PI       3.14159265358979323846
#define DOUBLE_EPS 1e-12

#include <device_launch_parameters.h>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/sort.h>


#define POWDI(x,i) pow(x,i)
#define GAUSSIAN 0
#define EXPONENTIAL 1
#define BISQUARE 2
#define TRICUBE 3
#define BOXCAR 4


// __global__ void coordinate_rotate(double* coords, int n, double costheta, double sintheta)
// {
//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     if (index >= n) return;
//     int i = index;
//     double ix = coords[i], iy = coords[i + n];
//     double ox = ix * costheta - iy * sintheta;
//     double oy = ix * sintheta + iy * costheta;
//     coords[i] = ox;
//     coords[i + n] = oy;
// }


// cudaError_t gw_coordinate_rotate_cuda(double* d_coords, int n, double theta, int threads)
// {
//     cudaError_t error;
//     dim3 blockSize(threads), gridSize((n + blockSize.x - 1) / blockSize.x);
//     coordinate_rotate << <gridSize, blockSize >> > (d_coords, n, cos(theta), sin(theta));
//     error = cudaGetLastError();
//     if (error != cudaSuccess)
//     {
//             return error;
//     }
//     return cudaSuccess;
// }


__global__ void eu_dist_vec_kernel(const double *dp, const double *rp, size_t rows, double *dists)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ndp) return;
    double ix = *(dp + i * 2), iy = *(dp + i * 2 + 1);
    double ox = *(rp), oy = *(rp + 1);
    double dist = hypot((ix - ox), (iy - oy));
    *(dists + i) = dist;
}

cudaError_t eu_dist_cuda(const double *d_dp, const double *d_rp, size_t rows, size_t threads, double* d_dists)
{
    cudaError_t error;
    dim3 blockSize(threads), gridSize((n + blockSize.x - 1) / blockSize.x);
    eu_dist_vec_kernel<<<gridSize, blockSize>>>(d_dp, d_rp, rows, d_dists);
    return cudaGetLastError();
}

__global__ void cd_dist_vec_kernel(const double *dp, int ndp, const double *rp, int focus, int nrp, double *dists)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= ndp) return;
    int i = index;
    double ix = dp[i], iy = dp[i + ndp];
    double ox = *(rp + focus), oy = *(rp + focus + nrp);
    double dist = fmax(fabs(ix - ox), fabs(iy - oy));
    *(dists + i) = dist;
}


__global__ void md_dist_vec_kernel(const double *dp, int ndp, const double *rp, int focus, int nrp, double *dists)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= ndp) return;
    int i = index;
    double ix = dp[i], iy = dp[i + ndp];
    double ox = *(rp + focus), oy = *(rp + focus + nrp);
    double dist = fabs(ix - ox) + fabs(iy - oy);
    *(dists + i) = dist;
}


__global__ void mk_dist_vec_kernel(const double *dp, int ndp, const double *rp, int focus, int nrp, double p, double *dists)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= ndp) return;
    int i = index;
    double ix = dp[i], iy = dp[i + ndp];
    double ox = *(rp + focus), oy = *(rp + focus + nrp);
    double dist = pow(pow(fabs(ix - ox), p) + pow(fabs(iy - oy), p), 1.0 / p);
    *(dists + i) = dist;
}

__device__ double sp_gcdist(double lon1, double lon2, double lat1, double lat2)
{
    double F, G, L, sinG2, cosG2, sinF2, cosF2, sinL2, cosL2, S, C;
    double w, R, a, f, D, H1, H2;
    double lat1R, lat2R, lon1R, lon2R, DE2RA;


    DE2RA = M_PI / 180;
    a = 6378.137;              /* WGS-84 equatorial radius in km */
    f = 1.0 / 298.257223563;     /* WGS-84 ellipsoid flattening factor */


    if (fabs(lat1 - lat2) < DOUBLE_EPS)
    {
        if (fabs(lon1 - lon2) < DOUBLE_EPS)
        {
            return 0.0;
            /* Wouter Buytaert bug caught 100211 */
        }
        else if (fabs((fabs(lon1) + fabs(lon2)) - 360.0) < DOUBLE_EPS)
        {
            return 0.0;
        }
    }
    lat1R = lat1 * DE2RA;
    lat2R = lat2 * DE2RA;
    lon1R = lon1 * DE2RA;
    lon2R = lon2 * DE2RA;


    F = (lat1R + lat2R) / 2.0;
    G = (lat1R - lat2R) / 2.0;
    L = (lon1R - lon2R) / 2.0;


    sinG2 = POWDI(sin(G), 2);
    cosG2 = POWDI(cos(G), 2);
    sinF2 = POWDI(sin(F), 2);
    cosF2 = POWDI(cos(F), 2);
    sinL2 = POWDI(sin(L), 2);
    cosL2 = POWDI(cos(L), 2);


    S = sinG2 * cosL2 + cosF2 * sinL2;
    C = cosG2 * cosL2 + sinF2 * sinL2;


    w = atan(sqrt(S / C));
    R = sqrt(S*C) / w;


    D = 2 * w*a;
    H1 = (3 * R - 1) / (2 * C);
    H2 = (3 * R + 1) / (2 * S);


    return D * (1 + f * H1*sinF2*cosG2 - f * H2*cosF2*sinG2);
}

__global__ void sp_dist_vec_kernel(const double *dp, int ndp, const double *rp, int focus, int nrp, double *dists)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= ndp) return;
    int i = index;
    double ix = dp[i], iy = dp[i + ndp];
    double ox = *(rp + focus), oy = *(rp + focus + nrp);
    dists[i] = sp_gcdist(ix, ox, iy, oy);
}

cudaError_t sp_dist_cuda(const double *d_dp, const double *d_rp, size_t rows, size_t threads, double* d_dists)
{
    cudaError_t error;
    dim3 blockSize(threads), gridSize((n + blockSize.x - 1) / blockSize.x);
    sp_dist_vec_kernel<<<gridSize, blockSize>>>(d_dp, d_rp, rows, d_dists);
    return cudaGetLastError();
}
