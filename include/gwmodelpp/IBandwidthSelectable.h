#ifndef IBANDWIDTHSELECTABLE_H
#define IBANDWIDTHSELECTABLE_H

#include "Status.h"
#include "spatialweight/BandwidthWeight.h"

namespace gwm
{

#define GWM_LOG_TAG_BANDWIDTH_CIRTERION "#bandwidth-criterion "

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
    static std::stringstream infoBandwidthCriterion(const BandwidthWeight* weight)
    {
        return std::stringstream() << GWM_LOG_TAG_BANDWIDTH_CIRTERION << (weight->adaptive() ? "adaptive" : "fixed") << "," << "criterion";
    }

    static std::stringstream infoBandwidthCriterion(const BandwidthWeight* weight, const double value)
    {
        if (weight->adaptive())
            return std::stringstream() << GWM_LOG_TAG_BANDWIDTH_CIRTERION << int(weight->bandwidth()) << "," << value;
        else 
            return std::stringstream() << GWM_LOG_TAG_BANDWIDTH_CIRTERION << weight->bandwidth() << "," << value;
    }

    /**
     * \~english
     * @brief Get criterion value with given bandwidth for bandwidth optimization.
     * 
     * @param weight Given bandwidth.
     * @param criterion [out] Criterion value.
     * @return Status Algorithm status.
     * 
     * \~chinese
     * @brief 根据指定的带宽计算带宽优选的指标值。
     * 
     * @param weight 指定的带宽。
     * @param criterion [出参] 带宽优选的指标值。
     * @param Status 算法运行状态。
     */
    virtual Status getCriterion(BandwidthWeight* weight, double& criterion) = 0;
};

typedef std::vector<std::pair<double, double> >  BandwidthCriterionList; //!< \~english A list of bandwidth criterions for all attempt bandwidth values. \~chinese 所有尝试的带宽对应的指标值列表


}

#endif  // IBANDWIDTHSELECTABLE_H