#include "gwmodelpp/spatialweight/CRSSTDistance.h"
#include <assert.h>

#include <exception>

using namespace std;
using namespace arma;
using namespace gwm;

vec CRSSTDistance::OrthogonalSTDistance(Distance* spatial, Distance* temporal, uword focus, double lambda, double angle)
{
    (void)angle;
    vec sdist = spatial->distance(focus);
    vec tdist = temporal->distance(focus);
    return lambda * sdist + (1 - lambda) * tdist;
}

vec CRSSTDistance::ObliqueSTDistance(Distance* spatial, Distance* temporal, uword focus, double lambda, double angle)
{
    vec sdist = spatial->distance(focus);
    vec tdist = temporal->distance(focus);
    return (lambda) * sdist + (1-lambda) * tdist + 2 * sqrt(lambda * (1 - lambda) * sdist * tdist) * cos(angle);
}

CRSSTDistance::CRSSTDistance() : 
    mSpatialDistance(nullptr),
    mTemporalDistance(nullptr),
    mLambda(0.0),
    mAngle(datum::pi / 2.0)
{
    mCalculator = &OrthogonalSTDistance;
}

CRSSTDistance::CRSSTDistance(Distance* spatialDistance, gwm::OneDimDistance* temporalDistance, double lambda) :
    mLambda(lambda),
    mAngle(datum::pi / 2.0)
{
    mSpatialDistance = spatialDistance->clone();
    mTemporalDistance = static_cast<gwm::OneDimDistance*>(temporalDistance->clone());
    mCalculator = &OrthogonalSTDistance;
}

CRSSTDistance::CRSSTDistance(Distance* spatialDistance, gwm::OneDimDistance* temporalDistance, double lambda, double angle) :
    mLambda(lambda),
    mAngle(angle)
{
    mSpatialDistance = spatialDistance->clone();
    mTemporalDistance = static_cast<gwm::OneDimDistance*>(temporalDistance->clone());
    mCalculator = (abs(angle - datum::pi / 2.0) < 1e-16) ? &OrthogonalSTDistance : &ObliqueSTDistance;
}

CRSSTDistance::CRSSTDistance(CRSSTDistance &distance)
{
    mLambda = distance.mLambda;
    mSpatialDistance = distance.mSpatialDistance->clone();
    mTemporalDistance = static_cast<gwm::OneDimDistance*>(distance.mTemporalDistance->clone());
}

void CRSSTDistance::makeParameter(initializer_list<DistParamVariant> plist)
{
    if (plist.size() == 4)
    {
        const mat& sfp = get<mat>(*(plist.begin()));
        const mat& sdp = get<mat>(*(plist.begin() + 1));
        const vec& tfp = get<vec>(*(plist.begin() + 2));
        const vec& tdp = get<vec>(*(plist.begin() + 3));
        if (sfp.n_rows == sdp.n_rows && sdp.n_rows == tfp.n_rows && tfp.n_rows == tdp.n_rows)
        {
            // mSpatialDistance->makeParameter(initializer_list<DistParamVariant>(plist.begin(), plist.begin() + 2));
            // mTemporalDistance->makeParameter(initializer_list<DistParamVariant>(plist.begin() + 2, plist.begin() + 4));
            // mParameter = make_unique<Parameter>();
            // mParameter->total = sfp.n_rows;
        }
        else
        {
            mParameter.reset(nullptr);
            throw std::runtime_error("Rows of points are not equal.");
        }
    }
    else
    {
        mParameter.reset(nullptr);
        throw std::runtime_error("The number of parameters must be 4.");
    }
}

double CRSSTDistance::maxDistance()
{
    if(mParameter == nullptr) throw std::runtime_error("Parameter is nullptr.");
    double maxD = 0.0;
    for (uword i = 0; i < mParameter->total; i++)
    {
        double d = max(distance(i));
        maxD = d > maxD ? d : maxD;
    }
    return maxD;
}

double CRSSTDistance::minDistance()
{
    if(mParameter == nullptr) throw std::runtime_error("Parameter is nullptr.");
    double minD = DBL_MAX;
    for (uword i = 0; i < mParameter->total; i++)
    {
        double d = min(distance(i));
        minD = d < minD ? d : minD;
    }
    return minD;
}