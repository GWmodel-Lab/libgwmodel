#include "gwmodelpp/spatialweight/CGwmCRSSTDistance.h"
#include <assert.h>

#include <exception>

vec CGwmCRSSTDistance::GcrsSTDistance(const rowvec& out_loc, const mat& in_locs ,double mLambda)
{
    uword N = in_locs.n_rows;
    vec STdists(N, fill::zeros);
    double uout = out_loc(0), vout = out_loc(1), tout=out_loc(2);
    double Sdist,Tdist;

    for (uword j = 0; j < N; j++)
    {
        Sdist = CGwmCRSDistance::SpGcdist(in_locs(j, 0), uout, in_locs(j, 1), vout);
        Tdist=in_locs(j, 2)-tout;
        //STdists(j) = sqrt((1-mLambda)*Sdist*Sdist+ mLambda *Tdist*Tdist);
        STdists(j) = (mLambda)*Sdist+(1-mLambda) *Tdist+2*sqrt(mLambda*(1-mLambda)*Sdist*Tdist);
    }
    return STdists;
}

vec CGwmCRSSTDistance::EuclideanDistance(const rowvec& out_loc, const mat& in_locs,double mLambda)
{
    uword N = in_locs.n_rows;
    vec STdists(N, fill::zeros);
    double uout = out_loc(0), vout = out_loc(1), tout=out_loc(2);
    double Sdist,Tdist,x,y;
    
    for (uword j = 0; j < N; j++)
    {
        x=in_locs(j, 0);
        y=in_locs(j, 1);
        x=abs(in_locs(j, 0)- uout);
        y=abs(in_locs(j, 1)- vout);
        Sdist = sqrt(x*x + y*y);
        Tdist = abs((in_locs(j, 2)-tout));
        //STdists(j) = sqrt((1-mLambda)*Sdist*Sdist+ mLambda *Tdist*Tdist);
        
        STdists(j) = (mLambda)*Sdist+(1-mLambda) *Tdist+2*sqrt(mLambda*(1-mLambda)*Sdist*Tdist);         
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
    mCalculator = mGeographic ? &GcrsSTDistance : &EuclideanDistance;
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
        return mCalculator(mParameter->focusPoints.row(focus), mParameter->dataPoints, mLambda);
    }
    else throw std::runtime_error("Target is out of bounds of data points.");
}




