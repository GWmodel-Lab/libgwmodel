#include "CGwmSpatialAlgorithm.h"

CGwmSpatialAlgorithm::CGwmSpatialAlgorithm()
{

}

CGwmSpatialAlgorithm::~CGwmSpatialAlgorithm()
{
    if (mSourceLayer) delete mSourceLayer;
    if (mResultLayer) delete mResultLayer;
}

bool CGwmSpatialAlgorithm::isValid()
{
    if (mSourceLayer == nullptr)
        return false;
    
    return true;
}