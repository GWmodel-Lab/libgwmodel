#include "gwmodelpp/spatialweight/MinkwoskiDistance.h"
#include <assert.h>

using namespace std;
using namespace arma;
using namespace gwm;

MinkwoskiDistance::MinkwoskiDistance(double p, double theta) : CRSDistance(false)
{
    mPoly = p;
    mTheta = theta;
}

MinkwoskiDistance::MinkwoskiDistance(const MinkwoskiDistance &distance) : CRSDistance(distance)
{
    mPoly = distance.mPoly;
    mTheta = distance.mTheta;
}

mat MinkwoskiDistance::CoordinateRotate(const mat& coords, double theta)
{
    uword n = coords.n_rows;
    mat rotated_coords(n, 2);
    rotated_coords.col(0) = coords.col(0) * cos(theta) - coords.col(1) * sin(theta);
    rotated_coords.col(1) = coords.col(0) * sin(theta) + coords.col(1) * cos(theta);
    return rotated_coords;
}

vec MinkwoskiDistance::distance(uword focus)
{
    if(mParameter == nullptr) throw std::runtime_error("Parameter is nullptr.");

    if (mGeographic) return CRSDistance::distance(focus);
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
            else if (mPoly == -1.0) return ManhattonDist(rp, dp);
            else return MinkwoskiDist(rp, dp, mPoly);
        }
        else throw std::runtime_error("Target is out of bounds of data points.");
    }
}
