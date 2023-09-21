#ifndef CRSDISTANCEKERNEL_H
#define CRSDISTANCEKERNEL_H

#include <cuda_runtime.h>

cudaError_t eu_dist_cuda(const double *d_dp, const double *d_rp, size_t rows, size_t threads, double* d_dists);
cudaError_t sp_dist_cuda(const double *d_dp, const double *d_rp, size_t rows, size_t threads, double* d_dists);

#endif  // CRSDISTANCEKERNEL_H
