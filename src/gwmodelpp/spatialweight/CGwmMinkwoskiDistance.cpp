#include "gwmodelpp/spatialweight/CGwmMinkwoskiDistance.h"

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

vec CGwmMinkwoskiDistance::distance(DistanceParameter* parameter, uword focus)
{
    _ASSERT(parameter != nullptr);
    if (mGeographic) return CGwmCRSDistance::distance(parameter, focus);
    else
    {
        CRSDistanceParameter* p = (CRSDistanceParameter*)parameter;
        if (p->dataPoints.n_cols == 2 && p->focusPoints.n_cols == 2)
        {
            if (focus < p->total)
            {
                mat dp = p->dataPoints;
                rowvec rp = p->focusPoints.row(focus);
                if (mPoly != 2 && mTheta != 0)
                {
                    dp = CoordinateRotate(p->dataPoints, mTheta);
                    rp = CoordinateRotate(p->focusPoints.row(focus), mTheta);
                }
                if (mPoly == 1.0) return ChessDistance(rp, dp);
                else if (mPoly == -1.0) return ManhattonDistance(rp, dp);
                else return MinkwoskiDistance(rp, dp, mPoly);
            }
            else throw std::runtime_error("Target is out of bounds of data points.");
        }
        else throw std::runtime_error("The dimension of data points or focus points is not 2.");
    }
}
