#ifndef CRSDISTANCEKERNEL_H
#define CRSDISTANCEKERNEL_H

#include <cuda_runtime.h>

/**
 * @brief \~english Calculate euclidean distance via CUDA. \~chinese 使用CUDA计算欧氏距离。
 * 
 * @param d_dp \~english Device pointer to data points, a matrix shaped $rows \times 2$ \~chinese 指向数据点矩阵的设备指针，形状为 $rows \times 2$
 * @param d_rp \~english Device pointer to the focus point, a matrix shaped $1 \times 2$ \~chinese 指向当前点矩阵的设备指针，形状为 $1 \times 2$
 * @param rows \~english The number of rows in `d_dp` \~chinese 矩阵 `d_dp` 的行数
 * @param threads \~english Number of GPU threads \~chinese GPU 线程数
 * @param d_dists \~english Output device pointer to distances \~chinese 输出距离的设备指针
 * @return cudaError_t \~english CUDA error or success \~chinese CUDA错误或者成功
 */
cudaError_t eu_dist_cuda(const double *d_dp, const double *d_rp, size_t rows, size_t threads, double* d_dists);

/**
 * @brief \~english Calculate geodetic distance via CUDA. \~chinese 使用CUDA计算测地线距离。
 * 
 * @param d_dp \~english Device pointer to data points, a matrix shaped $rows \times 2$ \~chinese 指向数据点矩阵的设备指针，形状为 $rows \times 2$
 * @param d_rp \~english Device pointer to the focus point, a matrix shaped $1 \times 2$ \~chinese 指向当前点矩阵的设备指针，形状为 $1 \times 2$
 * @param rows \~english The number of rows in `d_dp` \~chinese 矩阵 `d_dp` 的行数
 * @param threads \~english Number of GPU threads \~chinese GPU 线程数
 * @param d_dists \~english Output device pointer to distances \~chinese 输出距离的设备指针
 * @return cudaError_t \~english CUDA error or success \~chinese CUDA错误或者成功
 */
cudaError_t sp_dist_cuda(const double *d_dp, const double *d_rp, size_t rows, size_t threads, double* d_dists);

#endif  // CRSDISTANCEKERNEL_H
