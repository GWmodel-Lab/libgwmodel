#include "CGwmSpatialMultiscaleAlgorithm.h"

CGwmSpatialMultiscaleAlgorithm::CGwmSpatialMultiscaleAlgorithm()
{

}

CGwmSpatialMultiscaleAlgorithm::~CGwmSpatialMultiscaleAlgorithm()
{

}
/*
void CGwmSpatialMultiscaleAlgorithm::createDistanceParameter()
{
     for (uword i = 0; i < mIndepVars.size(); i++){
        if (mSpatialWeights[i].distance()->type() == CGwmDistance::DistanceType::CRSDistance || 
            mSpatialWeights[i].distance()->type() == CGwmDistance::DistanceType::MinkwoskiDistance)
        {
            mSpatialWeights[i].distance().makeParameter({
                mSourceLayer->points(),
                mSourceLayer->points()
            });
        }
    }
}
*/
