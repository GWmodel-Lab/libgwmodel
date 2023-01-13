#include "gwmodelpp/spatialweight/CGwmMinkwoskiDistance.h"
#include <assert.h>

using namespace std;
using namespace arma;

CGwmMinkwoskiDistance::CGwmMinkwoskiDistance(double p, double theta) : CGwmCRSDistance(false)
{
    mPoly = p;
    mTheta = theta;
}

CGwmMinkwoskiDistance::CGwmMinkwoskiDistance(const CGwmMinkwoskiDistance &distance) : CGwmCRSDistance(distance)
{
    mPoly = distance.mPoly;
    mTheta = distance.mTheta;
}

mat CGwmMinkwoskiDistance::CoordinateRotate(const mat& coords, double theta)
{
    uword n = coords.n_rows;
    mat rotated_coords(n, 2);
    rotated_coords.col(0) = coords.col(0) * cos(theta) - coords.col(1) * sin(theta);
    rotated_coords.col(1) = coords.col(0) * sin(theta) + coords.col(1) * cos(theta);
    return rotated_coords;
}

vec CGwmMinkwoskiDistance::distance(uword focus) const
{
    if(mParameter == nullptr) throw std::runtime_error("Parameter is nullptr.");

    if (mGeographic) return CGwmCRSDistance::distance(focus);
    else
    {
        if (focus < mParameter->total)
        {
            mat dp = mParameter->dataPoints;
            rowvec rp = mParameter->focusPoints.row(focus);
            if (mPoly != 2 && mTheta != 0)
            {
                dp = CoordinateRotate(mParameter->dataPoints, mTheta);
                rp = CoordinateRotate(mParameter->focusPoints.row(focus), mTheta);
            }
            if (mPoly == 1.0) return ChessDistance(rp, dp);
            else if (mPoly == -1.0) return ManhattonDistance(rp, dp);
            else return MinkwoskiDistance(rp, dp, mPoly);
        }
        else throw std::runtime_error("Target is out of bounds of data points.");
    }
}
