#ifndef CUDAUTILS_H
#define CUDAUTILS_H

#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

static const char *_cudaGetErrorEnum(cudaError_t error) {
    return cudaGetErrorName(error);
}


static const char *_cudaGetErrorEnum(cublasStatus_t error) {
    switch (error) {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
    }

    return "<unknown>";
}


#define DEVICE_RESET cudaDeviceReset();


template <typename T>
bool check(T result, char const *const func, const char *const file,
    int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
        DEVICE_RESET
        // Make sure we call CUDA Device Reset before exiting
        return true;
    }
    else return false;
}

#define checkCudaErrors(val) if (check((val), #val, __FILE__, __LINE__)) throw std::exception(std::runtime_error(_cudaGetErrorEnum(val)))

void pdm(const double* dptr, size_t rows, size_t cols, const char* header = "");

void pdc(const double* dptr, size_t rows, size_t cols, size_t strides, const char* header = "");

#endif  // CUDAUTILS_H
