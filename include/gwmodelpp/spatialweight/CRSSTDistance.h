#ifndef CRSSTDISTANCE_H
#define CRSSTDISTANCE_H

#include "CRSDistance.h"
#include "OneDimDistance.h"

namespace gwm
{

class CRSSTDistance : public OneDimDistance
{
public:
    typedef arma::vec (*CalculatorType)(Distance*, Distance*, arma::uword, double, double);

    static arma::vec OrthogonalSTDistance(Distance* spatial, Distance* temporal, arma::uword focus, double lambda, double angle);

    static arma::vec ObliqueSTDistance(Distance* spatial, Distance* temporal, arma::uword focus, double lambda, double angle);

public:
    CRSSTDistance();

    explicit CRSSTDistance(Distance* spatialDistance, OneDimDistance* temporalDistance, double lambda);

    explicit CRSSTDistance(Distance* spatialDistance, OneDimDistance* temporalDistance, double lambda, double angle);

    /**
     * @brief Copy construct a new CRSDistance object.
     * 
     * @param distance Refernce to object for copying.
     */
    CRSSTDistance(CRSSTDistance& distance);

    Distance * clone() override
    {
        return new CRSSTDistance(*this);
    }

    DistanceType type() const { return DistanceType::CRSSTDistance; }

    void makeParameter(std::initializer_list<DistParamVariant> plist) override;

    arma::vec distance(arma::uword focus) const
    {
        return mCalculator(mSpatialDistance, mTemporalDistance, focus, mLambda, mAngle);
    }

    double minDistance() const;

    double maxDistance() const;

public:

    const Distance* spatialDistance() const { return mSpatialDistance; }

    const OneDimDistance* temporalDistance() const { return mTemporalDistance; }

    double lambda() const { return mLambda; }

    void setLambda(const double lambda) { mLambda = lambda; }

protected:

    Distance* mSpatialDistance = nullptr;
    OneDimDistance* mTemporalDistance = nullptr;

    double mLambda = 0.0;
    double mAngle = arma::datum::pi / 2;

private:
    std::unique_ptr<Parameter> mParameter;
    CalculatorType mCalculator = &OrthogonalSTDistance;
};

}

#endif // CRSSTDISTANCE_H