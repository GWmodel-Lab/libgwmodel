#ifndef CRSSTDISTANCE_H
#define CRSSTDISTANCE_H

#include "CRSDistance.h"
#include "OneDimDistance.h"
#include <math.h>

namespace gwm
{

/**
 * @brief \~english Class for calculating spatial temporal distance. \~chinese 计算时空距离的类，由空间距离和时间距离组成
 */
class CRSSTDistance : public Distance
{
public:
    
    /**
     * @brief \~english parameters used in calculating. 
     * \~chinese 距离计算的输入参数类型：空间距离，时间距离，计算序号，λ，余弦值的角度（默认为π/2）
     */
    typedef arma::vec (*CalculatorType)(Distance*, gwm::OneDimDistance*, arma::uword, double, double);

    /**
     * @brief \~english Calculate temporal and spatial distance, Orthogonal Distance 
     * \~chinese 一般情况下的时空距离计算。正交时空距离
     * 
     * @param spatial \~english spatial distance of class gwm::Distance
     * \~chinese gwm::Distance类的空间距离。
     * @param temporal \~english temporal distance of class gwm::Distance
     * \~chinese gwm::Distance类的时间距离。
     * @param focus \~english the number of data to calculate
     * \~chinese 要计算的数据的序号。
     * @param lambda \~english lambda
     * \~chinese 时空距离的相对权重值，λ控制空间距离，1-λ控制时间距离。
     * @param angle \~english angle
     * \~chinese 参数取默认值。
     * @return arma::vec \~english Distance vector \~chinese 计算得到的距离向量
     */
    static arma::vec OrthogonalSTDistance(Distance* spatial, gwm::OneDimDistance* temporal, arma::uword focus, double lambda, double angle);

    /**
     * @brief \~english Calculate temporal and spatial distance with angle, Oblique Distance 
     * \~chinese 有angle情况下的时空距离计算，斜交时空距离
     * 
     * @param spatial \~english spatial distance of class gwm::Distance
     * \~chinese gwm::Distance类的空间距离。
     * @param temporal \~english temporal distance of class gwm::Distance
     * \~chinese gwm::Distance类的时间距离。
     * @param focus \~english the number of data to calculate
     * \~chinese 要计算的数据的序号。
     * @param lambda \~english lambda
     * \~chinese 时空距离的相对权重值，λ控制空间距离，1-λ控制时间距离。
     * @param angle \~english angle
     * \~chinese 默认值是π/2，给函数提供angle值，计算斜交时空距离
     * @return arma::vec \~english Distance vector \~chinese 计算得到的距离向量
     */
    static arma::vec ObliqueSTDistance(Distance* spatial, gwm::OneDimDistance* temporal, arma::uword focus, double lambda, double angle);

public:

    /**
     * @brief Construct.
     */
    CRSSTDistance();

    explicit CRSSTDistance(Distance* spatialDistance, gwm::OneDimDistance* temporalDistance, double lambda);

    explicit CRSSTDistance(Distance* spatialDistance, gwm::OneDimDistance* temporalDistance, double lambda, double angle);

    /**
     * @brief Copy construct.
     * 
     * @param distance Refernce to object for copying.
     */
    CRSSTDistance(CRSSTDistance& distance);

    Distance * clone() override
    {
        return new CRSSTDistance(*this);
    }

    DistanceType type() override { return DistanceType::CRSSTDistance; }

    /**
     * \~english @brief make the input data, initialize mParameter.
     * \~chinese @brief 将输入的数据初始化到mSpatialDistance, mTemporalDistance中，并初始化mParameter
     * @param plist \~english need to contain 4 items, mat, mat, vec, vec
     * \~chinese 需要是4个数据组成：mat：目标空间点，mat：数据空间点，vec：目标时间戳，vec：数据时间戳
     */
    void makeParameter(std::initializer_list<DistParamVariant> plist) override;

    /**
     * @brief \~english calculate distance
     * \~chinese 利用mCalculator计算距离
     * @param focus \~english focus     * \~chinese 第几个数据
     * @return mCalculator \~english already initialized OrthogonalSTDistance or ObliqueSTDistance 
     * \~chinese 距离计算器，二选一，是已被初始化的
     */
    arma::vec distance(arma::uword focus) override
    {
        return mCalculator(mSpatialDistance, mTemporalDistance, focus, mLambda, mAngle);
    }

    /**
     * @brief \~english bandwidth calculation. \~chinese 用于计算带宽
     */
    double minDistance() override;

    double maxDistance() override;

public:

    //const gwm::CRSDistance* spatialDistance() const { return mSpatialDistance; }

    const Distance* spatialDistance() const { return mSpatialDistance; }

    const gwm::OneDimDistance* temporalDistance() const { return mTemporalDistance; }

    // unused code to set lambda
    // double lambda() const { return mLambda; }
    void setLambda(const double lambda)    {
        if (lambda >= 0 && lambda <= 1)
        {
            mLambda = lambda;
        }
        else
            throw std::runtime_error("The lambda must be in [0,1].");
    }

protected:

    //gwm::CRSDistance* mSpatialDistance = nullptr;
    
    Distance* mSpatialDistance = nullptr;
    gwm::OneDimDistance* mTemporalDistance = nullptr;

    double mLambda = 0.0;
    double mAngle = arma::datum::pi / 2;

private:
    std::unique_ptr<Parameter> mParameter;
    CalculatorType mCalculator = &OrthogonalSTDistance;
};

}

#endif // CRSSTDISTANCE_H