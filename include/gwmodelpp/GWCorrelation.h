#ifndef GWCORRELATION_H
#define GWCORRELATION_H

#include "SpatialMonoscaleAlgorithm.h"
#include "IMultivariableAnalysis.h"
#include "IParallelizable.h"
#include "IBandwidthSelectable.h"

namespace gwm
{

class GWCorrelation
{
public:
    
    /**
     * @brief \~english Construct a new GWCorrelation object. \~chinese 构造一个新的 GWCorrelation 对象。
     * 
     */
    GWCorrelation() {}

    /**
     * @brief \~english Destroy the GWCorrelation object. \~chinese 销毁 GWCorrelation 对象。
     * 
     */
    ~GWCorrelation() {}
};

}

#endif  // GWCORRELATION_H