#ifndef CSPATIALALGORITHM_H
#define CSPATIALALGORITHM_H

#include <armadillo>
using namespace arma;

#include "CSimpleLayer.h"

namespace gwmodel
{

class CSpatialAlgorithm
{
public:
    CSpatialAlgorithm();
    ~CSpatialAlgorithm();

public:
    CSimpleLayer* sourceLayer() const;
    void setSourceLayer(CSimpleLayer* layer);
    
    CSimpleLayer* resultLayer() const;
    void setResultLayer(CSimpleLayer* layer);

public:
    virtual bool isValid() = 0;

private:
    CSimpleLayer* mSourceLayer = nullptr;
    CSimpleLayer* mResultLayer = nullptr;
};

inline CSimpleLayer* CSpatialAlgorithm::sourceLayer() const
{
    return mSourceLayer;
}

inline void CSpatialAlgorithm::setSourceLayer(CSimpleLayer* layer)
{
    mSourceLayer = layer;
}

inline CSimpleLayer* CSpatialAlgorithm::resultLayer() const
{
    return mResultLayer;
}

inline void CSpatialAlgorithm::setResultLayer(CSimpleLayer* layer)
{
    mResultLayer = layer;
}
    
} // namespace gwmodel


#endif  // CSPATIALALGORITHM_H