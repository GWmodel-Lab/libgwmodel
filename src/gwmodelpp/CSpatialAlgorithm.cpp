#include "CSpatialAlgorithm.h"

using namespace gwmodel;

CSpatialAlgorithm::CSpatialAlgorithm()
{

}

CSpatialAlgorithm::~CSpatialAlgorithm()
{
    if (mSourceLayer) delete mSourceLayer;
    if (mResultLayer) delete mResultLayer;
}

bool CSpatialAlgorithm::isValid()
{
    if (mSourceLayer == nullptr)
        return false;
    
    if (mResultLayer == nullptr)
        return false;
    
    return true;
}