#ifndef WEIGHT_H
#define WEIGHT_H

#include <unordered_map>
#include <string>
#include <armadillo>


namespace gwm
{

/**
 * @brief \~english Abstract base class for calculating weight from distance.
 * \~chinese 根据距离计算权重的基类。
 */
class Weight
{
public:

    /**
     * @brief \~english Type of weight. \~chinese 权重的类型。
     */
    enum WeightType
    {
        BandwidthWeight //!< \~english Bandwidth weight \~chinese 基于带宽的权重
    };

    static std::unordered_map<WeightType, std::string> TypeNameMapper;

public:

    /**
     * @brief \~english Construct a new Weight object. \~chinese 构造一个新的 Weight 对象。
     */
    Weight() {}

    /**
     * @brief \~english Destroy the Weight object. \~chinese 销毁 Weight 对象。
     */
    virtual ~Weight() {}

    /**
     * @brief \~english Clone this object. \~chinese 克隆该对象。
     * 
     * @return \~english Newly created pointer \~chinese 新创建的指针
     */
    virtual Weight* clone() = 0;

public:

    /**
     * @brief \~english Calculate weight vector from a distance vector.  \~chinese 从距离计算权重。
     * 
     * @param dist \~english According distance vector \~chinese 距离向量
     * @return \~english Weight vector \~chinese 权重向量
     */
    virtual arma::vec weight(arma::vec dist) = 0;
};

}

#endif // WEIGHT_H
