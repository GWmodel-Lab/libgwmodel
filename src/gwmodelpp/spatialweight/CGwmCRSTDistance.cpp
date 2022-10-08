#include "gwmodelpp/spatialweight/CGwmCRSTDistance.h"
#include <assert.h>

#include <exception>

vec CGwmCRSTDistance::SpatialTemporalDistance(const rowvec& out_loc, const mat& in_locs)
{
    uword N = in_locs.n_rows;
    vec STdists(N, fill::zeros);
    double uout = out_loc(0), vout = out_loc(1), tout=out_loc(2);
    double Sdist,Tdist;
    //double mLambda=0.05;
    CGwmCRSTDistance temproportion;
    for (uword j = 0; j < N; j++)
    {
        Sdist = CGwmCRSDistance::SpGcdist(in_locs(j, 0), uout, in_locs(j, 1), vout);
        Tdist=in_locs(j, 2)-tout;
        //STdists(j) = sqrt((1-mLambda)*Sdist*Sdist+ mLambda *Tdist*Tdist);
        STdists(j) = sqrt((1-temproportion.mLambda)*Sdist*Sdist+ temproportion.mLambda *Tdist*Tdist);
    }
    return STdists;
}

CGwmCRSTDistance::CGwmCRSTDistance() : mParameter(nullptr)
{

}

CGwmCRSTDistance::CGwmCRSTDistance(bool isGeographic, double mLambda): mParameter(nullptr), mGeographic(isGeographic)
{
    mCalculator = mGeographic ? &SpatialTemporalDistance : &EuclideanDistance;
}

CGwmCRSTDistance::CGwmCRSTDistance(const CGwmCRSTDistance &distance) : CGwmCRSDistance(distance)
{
    mGeographic = distance.mGeographic;
    if (distance.mParameter)
    {
        mat fp = distance.mParameter->focusPoints;
        mat dp = distance.mParameter->dataPoints;
        mParameter = make_unique<Parameter>(fp, dp);
    }
}

void CGwmCRSTDistance::makeParameter(initializer_list<DistParamVariant> plist)
{
    if (plist.size() == 2)
    {
        const mat& fp = get<mat>(*(plist.begin()));
        const mat& dp = get<mat>(*(plist.begin() + 1));
        if (fp.n_cols == 2 && dp.n_cols == 2)
            mParameter = make_unique<Parameter>(fp, dp);
        else 
        {
            mParameter.reset(nullptr);
            throw std::runtime_error("The dimension of data points or focus points is not 2."); 
        }
    }
    else
    {
        mParameter.reset(nullptr);
        throw std::runtime_error("The number of parameters must be 2.");
    }
}

vec CGwmCRSTDistance::distance(uword focus)
{
    if(mParameter == nullptr) throw std::runtime_error("Parameter is nullptr.");

    if (focus < mParameter->total)
    {
        return mCalculator(mParameter->focusPoints.row(focus), mParameter->dataPoints);
    }
    else throw std::runtime_error("Target is out of bounds of data points.");
}

double CGwmCRSTDistance::maxDistance()
{
    if(mParameter == nullptr) throw std::runtime_error("Parameter is nullptr.");
    double maxD = 0.0;
    for (uword i = 0; i < mParameter->total; i++)
    {
        double d = max(mCalculator(mParameter->focusPoints.row(i), mParameter->dataPoints));
        maxD = d > maxD ? d : maxD;
    }
    return maxD;
}

double CGwmCRSTDistance::minDistance()
{
    if(mParameter == nullptr) throw std::runtime_error("Parameter is nullptr.");
    double minD = DBL_MAX;
    for (uword i = 0; i < mParameter->total; i++)
    {
        double d = min(mCalculator(mParameter->focusPoints.row(i), mParameter->dataPoints));
        minD = d < minD ? d : minD;
    }
    return minD;
}



