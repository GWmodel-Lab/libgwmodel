#include "CGwmSpatialAlgorithm.h"

using namespace gwm;

bool CGwmSpatialAlgorithm::isValid()
{
    return !mCoords.is_empty();
}