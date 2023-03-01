#ifndef IBANDWIDTHSELECTABLE_H
#define IBANDWIDTHSELECTABLE_H

#include "spatialweight/BandwidthWeight.h"

namespace gwm
{

/**
 * \~english
 * @brief Interface for bandwidth-selectable algorithm.
 * 
 * \~chinese
 * @brief 可选带宽算法接口。
 * 
 */
struct IBandwidthSelectable
{
    /**
     * \~english
     * @brief Get criterion value with given bandwidth for bandwidth optimization.
     * 
     * @param weight Given bandwidth
     * @return double Criterion value
     * 
     * \~chinese
     * @brief 根据指定的带宽计算带宽优选的指标值。
     * 
     * @param weight 指定的带宽。
     * @return double 带宽优选的指标值。
     */
    virtual double getCriterion(BandwidthWeight* weight) = 0;
};

typedef std::vector<std::pair<double, double> >  BandwidthCriterionList; //!< \~english A list of bandwidth criterions for all attempt bandwidth values. \~chinese 所有尝试的带宽对应的指标值列表


}

#endif  // IBANDWIDTHSELECTABLE_H