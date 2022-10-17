#include "CGwmSpatialMultiscaleAlgorithm.h"

CGwmSpatialMultiscaleAlgorithm::CGwmSpatialMultiscaleAlgorithm()
{

}

CGwmSpatialMultiscaleAlgorithm::~CGwmSpatialMultiscaleAlgorithm()
{

}

void CGwmSpatialMultiscaleAlgorithm::createDistanceParameter(size_t size)
{
     for (uword i = 0; i < size; i++){
        if (mSpatialWeights[i].distance()->type() == CGwmDistance::DistanceType::CRSDistance || 
            mSpatialWeights[i].distance()->type() == CGwmDistance::DistanceType::MinkwoskiDistance)
        {
            mSpatialWeights[i].distance()->makeParameter({
                mSourceLayer->points(),
                mSourceLayer->points()
            });
        }
    }
}
