#ifndef CGWMSPATIALALGORITHM_H
#define CGWMSPATIALALGORITHM_H

#include <armadillo>
using namespace arma;

#include "CGwmSimpleLayer.h"

class CGwmSpatialAlgorithm
{
public:
    CGwmSpatialAlgorithm();
    ~CGwmSpatialAlgorithm();

public:
    CGwmSimpleLayer* sourceLayer() const;
    void setSourceLayer(CGwmSimpleLayer* layer);
    
    CGwmSimpleLayer* resultLayer() const;
    void setResultLayer(CGwmSimpleLayer* layer);

public:
    virtual bool isValid() = 0;

protected:
    CGwmSimpleLayer* mSourceLayer = nullptr;
    CGwmSimpleLayer* mResultLayer = nullptr;
};

inline CGwmSimpleLayer* CGwmSpatialAlgorithm::sourceLayer() const
{
    return mSourceLayer;
}

inline void CGwmSpatialAlgorithm::setSourceLayer(CGwmSimpleLayer* layer)
{
    mSourceLayer = layer;
}

inline CGwmSimpleLayer* CGwmSpatialAlgorithm::resultLayer() const
{
    return mResultLayer;
}

inline void CGwmSpatialAlgorithm::setResultLayer(CGwmSimpleLayer* layer)
{
    mResultLayer = layer;
}


#endif  // CGWMSPATIALALGORITHM_H