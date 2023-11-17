#ifndef DISTANCE_H
#define DISTANCE_H

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include "gwmodelpp/spatialweight/cuda/ISpatialCudaEnabled.h"
#endif // ENABLE_CUDA

#include <memory>
#include <string>
#include <unordered_map>
#include <armadillo>
#include <variant>


namespace gwm
{

typedef std::variant<arma::mat, arma::vec, arma::uword> DistParamVariant;   //!< \~english \~english Acceptable types of distance paramters. \~chinese 可接受的距离参数类型。

/**
 * @brief \~english Abstract base class for calculating spatial distance. \~chinese 空间距离度量基类。
 * 
 */
class Distance
#ifdef ENABLE_CUDA
    : public ISpatialCudaEnabled
#endif
{
public:

    /**
     * @brief \~english Struct of parameters used in spatial distance calculating. 
     * Usually a pointer to object of its derived classes is passed to Distance::distance().
     * \~chinese 距离计算用的参数。通常将派生类指针传递到 Distance::distance() 中。
     */
    struct Parameter
    {
        arma::uword total;    //!< \~english Total data points \~chinese 数据点总数

        /**
         * @brief \~english Construct a new DistanceParameter object. \~chinese 构造一个新的 DistanceParameter 对象。
         */
        Parameter(): total(0) {}
    };

    /**
     * @brief \~english Types of distance. \~chinese 距离度量类型。
     */
    enum DistanceType
    {
        CRSDistance,        //!< \~english Distance according to coordinate reference system \~chinese 坐标系距离
        MinkwoskiDistance,  //!< \~english Minkwoski distance \~chinese Minkwoski 距离
        DMatDistance,       //!< \~english Distance according to a .dmat file \~chinese 从 .dmat 文件读取距离
        OneDimDistance,     //!< \~english Distance for just one dimension \~chinese 一维距离
        CRSSTDistance,
    };
    
    /**
     * @brief \~english A mapper between types of distance and its names. \~chinese 距离度量类型和名称映射表。
     * 
     */
    static std::unordered_map<DistanceType, std::string> TypeNameMapper;

public:

    /**
     * @brief \~english Destroy the Distance object. \~chinese 销毁 Distance 对象。
     */
    virtual ~Distance() {};

    /**
     * @brief \~english Clone this Distance object. \~chinese 克隆这个 Distance 对象。
     * 
     * @return Distance* \~english Newly created pointer \~chinese 重新创建的对象指针
     */
    virtual Distance* clone() const = 0;

    /**
     * @brief \~english Return the type of this object. \~chinese 返回该对象的类型。
     * 
     * @return DistanceType \~english Type of distance \~chinese 距离陆良类型
     */
    virtual DistanceType type() = 0;

    virtual Parameter* parameter() const = delete;


public:

    /**
     * @brief \~english Create Parameter for Caclulating Distance.
     * This function is pure virtual. It would never be called directly.
     * \~chinese 创建用于计算距离的参数。该函数为纯虚函数。
     * 
     * @param plist \~english A list of parameters \~chinese 
     */
    virtual void makeParameter(std::initializer_list<DistParamVariant> plist) = 0;

    /**
     * @brief \~english Calculate distance vector for a focus point. \~chinese 为一个目标点计算距离向量。
     * 
     * @param focus \~english Focused point's index. Require focus < total \~chinese 目标点索引，要求 focus 小于参数中的 total
     * @return arma::vec \~english Distance vector for the focused point \~chinese 目标点到所有数据点的距离向量
     */
    virtual arma::vec distance(arma::uword focus) = 0;

#ifdef ENABLE_CUDA

    virtual bool useCuda() override { return mUseCuda; }

    virtual void setUseCuda(bool isUseCuda) override { mUseCuda = isUseCuda; }

    virtual cudaError_t prepareCuda(size_t gpuId) override;
    
    /**
     * @brief \~english Calculate distance vector for a focus point. \~chinese 为一个目标点计算距离向量。
     * 
     * @param focus \~english Focused point's index. Require focus < total \~chinese 目标点索引，要求 focus 小于参数中的 total
     * @param d_dists \~english Output device pointer to distances \~chinese 指向输出距离的设备指针
     * @param elems \~english Number of elements in distances \~chinese 距离向量的元素数量
     * @return cudaError_t \~english CUDA error or success \~chinese CUDA 错误或成功
     */
    virtual cudaError_t distance(arma::uword focus, double* d_dists, size_t* elems)
    {
        throw std::logic_error("Function not yet implemented");
    }

#endif // ENABLE_CUDA

    /**
     * @brief \~english Get maximum distance among all points. \~chinese 获取最大距离。
     * 
     * @return double \~english Maximum distance \~chinese 最大距离
     */
    virtual double maxDistance() = 0;
    
    /**
     * @brief \~english Get minimum distance among all points \~chinese 获取最小距离。
     * 
     * @return double \~english Maximum distance \~chinese 最小距离
     */
    virtual double minDistance() = 0;

#ifdef ENABLE_CUDA
protected:
    bool mUseCuda = false;  //<! \~english Whether to use CUDA \~chinese 是否使用 CUDA
    int mGpuID = 0;  //<! \~english The ID of selected GPU \~chinese 选择的 GPU 的索引
    bool mCudaPrepared = false;  //<! \~english Whether CUDA has been prepared \~chinese CUDA 环境是否已经准备
    size_t mCudaThreads = 0;  //<! \~english Number of GPU threads \~chinese GPU 线程数

#endif // ENABLE_CUDA

};

}


#endif // DISTANCE_H
