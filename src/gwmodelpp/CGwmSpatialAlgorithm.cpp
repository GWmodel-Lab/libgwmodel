#include "CGwmSpatialAlgorithm.h"

CGwmSpatialAlgorithm::CGwmSpatialAlgorithm()
{

}

CGwmSpatialAlgorithm::~CGwmSpatialAlgorithm()
{
    if (mSourceLayer != nullptr)
        delete mSourceLayer;
}

bool CGwmSpatialAlgorithm::isValid()
{
    if (mSourceLayer == nullptr)
        return false;
    
    return true;
}