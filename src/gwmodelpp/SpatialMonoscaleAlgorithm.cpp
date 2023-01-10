#include "SpatialMonoscaleAlgorithm.h"

using namespace gwm;

void SpatialMonoscaleAlgorithm::createDistanceParameter()
{
    if (mSpatialWeight.distance()->type() == Distance::DistanceType::CRSDistance || 
        mSpatialWeight.distance()->type() == Distance::DistanceType::MinkwoskiDistance)
    {
        mSpatialWeight.distance()->makeParameter({ mCoords, mCoords });
    }
}
