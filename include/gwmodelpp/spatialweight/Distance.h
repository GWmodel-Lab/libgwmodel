#ifndef DISTANCE_H
#define DISTANCE_H

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
    virtual Distance* clone() = 0;

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

};

}


#endif // DISTANCE_H
