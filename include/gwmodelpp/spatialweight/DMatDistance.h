#ifndef DMATDISTANCE_H
#define DMATDISTANCE_H

#include <string>
#include "Distance.h"


namespace gwm
{

/**
 * [NOT AVALIABLE]
 */
class DMatDistance : public Distance
{
public:

    /**
     * @brief \~english Struct of parameters used in spatial distance calculating according to coordinate reference system. 
     * \~chinese 距离计算用的参数。
     */
    struct Parameter : public Distance::Parameter
    {
        arma::uword rowSize;    //!< \~english Size of each rows \~chinese 每行的大小

        Parameter(arma::uword size, arma::uword rows) : rowSize(size) 
        {
            total = rows;
        }
    };

public:
    /**
     * @brief \~english Construct a new DMatDistance object. \~chinese 构造新的 DMatDistance 对象。
     * 
     * @param dmatFile \~english Path to file of distance matrix \~chinese 距离矩阵文件路径
     */
    explicit DMatDistance(std::string dmatFile);

    /**
     * @brief \~english Copy construct a new DMatDistance object. \~chinese 复制构造新的 {name} 对象。
     * 
     * @param distance \~english DMatDistance object \~chinese DMatDistance 对象
     */
    DMatDistance(const DMatDistance& distance);

    virtual Distance * clone() const override
    {
        return new DMatDistance(*this);
    }

    DistanceType type() override { return DistanceType::DMatDistance; }

    /**
     * @brief \~english Get the path to DMat file \~chinese 获取 DMat 文件的路径
     * 
     * @return std::string \~english Path to DMat file \~chinese DMat 文件的路径
     */
    std::string dMatFile() const;

    /**
     * @brief \~english Set the path to DMat file \~chinese 设置 DMat 文件的路径
     * 
     * @param dMatFile \~english Path to DMat file \~chinese DMat 文件的路径
     */
    void setDMatFile(const std::string &dMatFile);

public:

    /**
     * @brief Create Parameter for Caclulating CRS Distance.
     * 
     * @param plist A list of parameters containing 2 items:
     *  - `arma::uword` size
     *  - `arma::uword` rows
     *  . 
     * 
     * @return DistanceParameter* The pointer to parameters.
     */
    virtual void makeParameter(std::initializer_list<DistParamVariant> plist) override;
    
    virtual arma::vec distance(arma::uword focus) override;
    virtual double maxDistance() override;
    virtual double minDistance() override;

private:
    std::string mDMatFile;  //!< \~english Path to a file of distance matrix \~chinese 距离矩阵文件的路径
    std::unique_ptr<Parameter> mParameter = nullptr;  //!< \~english Parameter \~chinese 参数
};

inline std::string DMatDistance::dMatFile() const
{
    return mDMatFile;
}

inline void DMatDistance::setDMatFile(const std::string &dMatFile)
{
    mDMatFile = dMatFile;
}

}

#endif // DMATDISTANCE_H
