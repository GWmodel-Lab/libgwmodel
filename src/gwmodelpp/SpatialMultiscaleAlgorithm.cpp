#include "SpatialMultiscaleAlgorithm.h"

using namespace std;
using namespace arma;
using namespace gwm;

void SpatialMultiscaleAlgorithm::createDistanceParameter(size_t size)
{
     for (uword i = 0; i < size; i++){
        if (mSpatialWeights[i].distance()->type() == Distance::DistanceType::CRSDistance || 
            mSpatialWeights[i].distance()->type() == Distance::DistanceType::MinkwoskiDistance)
        {
            mSpatialWeights[i].distance()->makeParameter({ mCoords, mCoords });
        }
    }
}
