#ifndef MINKWOSKIDISTANCE_H
#define MINKWOSKIDISTANCE_H

#include "CRSDistance.h"

namespace gwm
{

/**
 * @brief \~english Minkwoski distnace \~chinese 明氏距离
 * 
 */
class MinkwoskiDistance : public CRSDistance
{
public:

    /**
     * @brief \~english Rotate coordiantes. \~chinese 旋转坐标。
     * 
     * @param coords \~english Coordiantes \~chinese 坐标
     * @param theta \~english Angle \~chinese 旋转角度
     * @return arma::mat \~english Rotated coordinates \~chinese 旋转后的坐标
     */
    static arma::mat CoordinateRotate(const arma::mat& coords, double theta);

    /**
     * @brief \~english Chess distance. \~chinese 棋盘距离。
     * 
     * @param out_loc \~english Coordinate of focus point \~chinese 目标点坐标
     * @param in_locs \~english Coordinates of data poitnts \~chinese 数据点坐标
     * @return arma::vec \~english Distance vector from the focus point to data points \~chinese 目标点到数据点距离向量
     */
    static arma::vec ChessDistance(const arma::rowvec& out_loc, const arma::mat& in_locs);

    /**
     * @brief \~english Manhatton distance. \~chinese 曼哈顿距离。
     * 
     * @param out_loc \~english Coordinate of focus point \~chinese 目标点坐标
     * @param in_locs \~english Coordinates of data poitnts \~chinese 数据点坐标
     * @return arma::vec \~english Distance vector from the focus point to data points \~chinese 目标点到数据点距离向量
     */
    static arma::vec ManhattonDist(const arma::rowvec& out_loc, const arma::mat& in_locs);

    /**
     * @brief \~english Minkwoski distnace \~chinese 明氏距离
     * 
     * @param out_loc \~english Coordinate of focus point \~chinese 目标点坐标
     * @param in_locs \~english Coordinates of data poitnts \~chinese 数据点坐标
     * @param p \~english Polynomial number \~chinese 次数
     * @return arma::vec \~english Distance vector from the focus point to data points \~chinese 目标点到数据点距离向量
     */
    static arma::vec MinkwoskiDist(const arma::rowvec& out_loc, const arma::mat& in_locs, double p);

public:

    MinkwoskiDistance() : mPoly(2.0), mTheta(0.0) {}

    /**
     * @brief \~english Construct a new MinkwoskiDistance object \~chinese 构造一个新的 MinkwoskiDistance 对象
     * 
     * @param p \~english Polynomial number \~chinese 次数
     * @param theta \~english Angle \~chinese 旋转角度
     */
    MinkwoskiDistance(double p, double theta);

    /**
     * @brief \~english Copy construct a new MinkwoskiDistance object \~chinese 构造一个新的 MinkwoskiDistance 对象
     * 
     * @param distance \~english The MinkwoskiDistance object to be copied \~chinese 要拷贝的 MinkwoskiDistance 对象
     */
    MinkwoskiDistance(const MinkwoskiDistance& distance);

    virtual Distance * clone() override
    {
        return new MinkwoskiDistance(*this);
    }

    DistanceType type() override { return DistanceType::MinkwoskiDistance; }

    /**
     * @brief \~english Get the polynomial number. \~chinese 获取次数。
     * 
     * @return double \~english Polynomial number \~chinese 次数 
     */
    double poly() const;

    /**
     * @brief \~english Set the polynomial number. \~chinese 设置次数。
     * 
     * @param poly \~english Polynomial number \~chinese 次数
     */
    void setPoly(double poly);

    /**
     * @brief \~english Get the angle. \~chinese 获取旋转角度。
     * 
     * @return double \~english Angle \~chinese 旋转角度 
     */
    double theta() const;

    /**
     * @brief \~english Set the angle. \~chinese 设置旋转角度。
     * 
     * @param theta \~english Angle \~chinese 旋转角度
     */
    void setTheta(double theta);

public:
    virtual arma::vec distance(arma::uword focus) override;

private:
    double mPoly = 2.0;
    double mTheta = 0.0;
};

inline arma::vec MinkwoskiDistance::ChessDistance(const arma::rowvec& out_loc, const arma::mat& in_locs)
{
    return max(abs(in_locs.each_row() - out_loc), 1);
}

inline arma::vec MinkwoskiDistance::ManhattonDist(const arma::rowvec& out_loc, const arma::mat& in_locs)
{
    return sum(abs(in_locs.each_row() - out_loc), 1);
}

inline arma::vec MinkwoskiDistance::MinkwoskiDist(const arma::rowvec& out_loc, const arma::mat& in_locs, double p)
{
    arma::vec temp = abs(in_locs.each_row() - out_loc);
    return pow(sum(pow(temp, p), 1), 1.0 / p);
}

inline double MinkwoskiDistance::poly() const
{
    return mPoly;
}

inline void MinkwoskiDistance::setPoly(double poly)
{
    mPoly = poly;
}

inline double MinkwoskiDistance::theta() const
{
    return mTheta;
}

inline void MinkwoskiDistance::setTheta(double theta)
{
    mTheta = theta;
}

}

#endif // MINKWOSKIDISTANCE_H
