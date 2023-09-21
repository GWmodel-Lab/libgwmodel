#ifndef BANDWIDTHWEIGHTKERNEL_H
#define BANDWIDTHWEIGHTKERNEL_H

#include <cuda_runtime.h>

cudaError_t gw_weight_cuda(double bw, int kernel, bool adaptive, double *d_dists, double *d_weight, int ndp, int threads);

#endif  // BANDWIDTHWEIGHTKERNEL_H
