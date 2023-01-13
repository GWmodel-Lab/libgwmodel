#ifndef CGWMCRSSTDISTANCE_H
#define CGWMCRSSTDISTANCE_H

#include "CGwmCRSDistance.h"
#include "CGwmOneDimDistance.h"

class CGwmCRSSTDistance : public CGwmDistance
{
public:
    typedef arma::vec (*CalculatorType)(const CGwmDistance*, const CGwmDistance*, arma::uword, double, double);

    static arma::vec OrthogonalSTDistance(const CGwmDistance* spatial, const CGwmDistance* temporal, arma::uword focus, double lambda, double angle);

    static arma::vec ObliqueSTDistance(const CGwmDistance* spatial, const CGwmDistance* temporal, arma::uword focus, double lambda, double angle);

public:
    CGwmCRSSTDistance();

    explicit CGwmCRSSTDistance(const CGwmDistance* spatialDistance, const CGwmOneDimDistance* temporalDistance, double lambda);

    explicit CGwmCRSSTDistance(const CGwmDistance* spatialDistance, const CGwmOneDimDistance* temporalDistance, double lambda, double angle);

    /**
     * @brief Copy construct a new CGwmCRSDistance object.
     * 
     * @param distance Refernce to object for copying.
     */
    CGwmCRSSTDistance(const CGwmCRSSTDistance& distance);

    CGwmDistance * clone() const override
    {
        return new CGwmCRSSTDistance(*this);
    }

    DistanceType type() const override { return DistanceType::CRSSTDistance; }

    void makeParameter(std::initializer_list<DistParamVariant> plist) override;

    arma::vec distance(arma::uword focus) const override
    {
        return mCalculator(mSpatialDistance, mTemporalDistance, focus, mLambda, mAngle);
    }

    double minDistance() const override;

    double maxDistance() const override;

public:

    const CGwmDistance* spatialDistance() const { return mSpatialDistance; }

    const CGwmOneDimDistance* temporalDistance() const { return mTemporalDistance; }

    double lambda() const { return mLambda; }

    void setLambda(const double lambda) { mLambda = lambda; }

protected:

    CGwmDistance* mSpatialDistance = nullptr;
    CGwmOneDimDistance* mTemporalDistance = nullptr;

    double mLambda = 0.0;
    double mAngle = arma::datum::pi / 2;

private:
    std::unique_ptr<Parameter> mParameter;
    CalculatorType mCalculator = &OrthogonalSTDistance;
};


#endif // CGWMCRSSTDISTANCE_H