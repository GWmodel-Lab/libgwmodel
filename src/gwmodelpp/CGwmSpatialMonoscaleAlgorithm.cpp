#include "CGwmSpatialMonoscaleAlgorithm.h"

using namespace gwm;

void CGwmSpatialMonoscaleAlgorithm::createDistanceParameter()
{
    if (mSpatialWeight.distance()->type() == CGwmDistance::DistanceType::CRSDistance || 
        mSpatialWeight.distance()->type() == CGwmDistance::DistanceType::MinkwoskiDistance)
    {
        mSpatialWeight.distance()->makeParameter({ mCoords, mCoords });
    }
}
