#ifndef BANDWIDTHWEIGHTKERNEL_H
#define BANDWIDTHWEIGHTKERNEL_H

#include <cuda_runtime.h>

/**
 * @brief \~english Calculate spatial weights via CUDA. \~chinese 使用CUDA计算空间权重。
 * 
 * @param bw \~english Bandwidth size \~chinese 带宽大小
 * @param kernel \~english Kernel function index \~chinese 核函数索引
 * @param adaptive \~english Whether the bandwidth is adaptive \~chinese 是否是可变带宽
 * @param d_dists \~english Vector of input distance \~chinese 输入距离向量
 * @param d_weight \~english Vector of output weights \~chinese 输出权重向量
 * @param ndp \~english Number of data points \~chinese 总点数
 * @param threads \~english Number of GPU threads \~chinese GPU线程数
 * @return cudaError_t \~english CUDA error or success \~chinese CUDA错误或成功
 */
cudaError_t gw_weight_cuda(double bw, int kernel, bool adaptive, double *d_dists, double *d_weight, int ndp, int threads);

#endif  // BANDWIDTHWEIGHTKERNEL_H
