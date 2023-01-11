#include "gwmodelpp/spatialweight/CGwmCRSSTDistance.h"
#include <assert.h>

#include <exception>

vec CGwmCRSSTDistance::SpatialTemporalDistance(const rowvec& out_loc, const mat& in_locs)
{
    uword N = in_locs.n_rows;
    vec STdists(N, fill::zeros);
    double uout = out_loc(0), vout = out_loc(1), tout=out_loc(2);
    double Sdist,Tdist;
    //double mLambda=0.05;

    CGwmCRSSTDistance temproportion;
    for (uword j = 0; j < N; j++)
    {
        Sdist = CGwmCRSDistance::SpGcdist(in_locs(j, 0), uout, in_locs(j, 1), vout);
        Tdist=in_locs(j, 2)-tout;
        //STdists(j) = sqrt((1-mLambda)*Sdist*Sdist+ mLambda *Tdist*Tdist);
        STdists(j) = (temproportion.mLambda)*Sdist+(1-temproportion.mLambda) *Tdist+2*sqrt(temproportion.mLambda*(1-temproportion.mLambda)*Sdist*Tdist);
    }
    return STdists;
}

vec CGwmCRSSTDistance::EuclideanDistance(const rowvec& out_loc, const mat& in_locs)
{
    uword N = in_locs.n_rows;
    vec STdists(N, fill::zeros);
    double uout = out_loc(0), vout = out_loc(1), tout=out_loc(2);
    double Sdist,Tdist,x,y;
    CGwmCRSSTDistance temproportion;
    for (uword j = 0; j < N; j++)
    {
        x=abs(in_locs(j, 0)- uout);
        y=abs(in_locs(j, 1)- vout);
        Sdist = sqrt(x*x + y*y);
        Tdist = abs((in_locs(j, 2)-tout));
        //STdists(j) = sqrt((1-mLambda)*Sdist*Sdist+ mLambda *Tdist*Tdist);
        //x=sqrt((1-temproportion.mLambda)*Sdist+ temproportion.mLambda *Tdist*Tdist);
        
        //STdists(j) = sqrt((1-temproportion.mLambda)*Sdist*Sdist+ temproportion.mLambda *Tdist*Tdist);
        STdists(j) = (temproportion.mLambda)*Sdist+(1-temproportion.mLambda) *Tdist+2*sqrt(temproportion.mLambda*(1-temproportion.mLambda)*Sdist*Tdist);
    }
    return STdists;
    // mat diff = (in_locs.each_row() - out_loc);
    // return sqrt(sum(diff % diff, 1));
}

CGwmCRSSTDistance::CGwmCRSSTDistance() : mParameter(nullptr)
{

}

CGwmCRSSTDistance::CGwmCRSSTDistance(bool isGeographic, double lambda): mParameter(nullptr), mGeographic(isGeographic)
{
    mLambda=lambda;
    mCalculator = mGeographic ? &SpatialTemporalDistance : &EuclideanDistance;
}

CGwmCRSSTDistance::CGwmCRSSTDistance(const CGwmCRSSTDistance &distance) : CGwmCRSDistance(distance)
{
    mGeographic = distance.mGeographic;
    if (distance.mParameter)
    {
        mat fp = distance.mParameter->focusPoints;
        mat dp = distance.mParameter->dataPoints;
        mParameter = make_unique<Parameter>(fp, dp);
    }
}

void CGwmCRSSTDistance::makeParameter(initializer_list<DistParamVariant> plist)
{
    if (plist.size() == 2)
    {
        const mat& fp = get<mat>(*(plist.begin()));
        const mat& dp = get<mat>(*(plist.begin() + 1));
        if (fp.n_cols == 3 && dp.n_cols == 3 )
            mParameter = make_unique<Parameter>(fp, dp);
        else 
        {
            mParameter.reset(nullptr);
            throw std::runtime_error("The dimension of data points or focus points is not 3, maybe do not have timestamps."); 
        }
    }
    else
    {
        mParameter.reset(nullptr);
        throw std::runtime_error("The number of parameters must be 2 coordinate position with 1 timestamp.");
    }
}

vec CGwmCRSSTDistance::distance(uword focus)
{
    if(mParameter == nullptr) throw std::runtime_error("Parameter is nullptr.");

    if (focus < mParameter->total)
    {
        return mCalculator(mParameter->focusPoints.row(focus), mParameter->dataPoints);
    }
    else throw std::runtime_error("Target is out of bounds of data points.");
}




