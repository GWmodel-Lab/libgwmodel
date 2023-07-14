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

    /**
     * @brief \~english Get meta infomation of current bandwidth value and the corresponding criterion value.
     * \~chinese 获取当前带宽值和对应指标值的元信息。
     * 
     * @param weight \~english Bandwidth weight \~chinese 带宽设置
     * @return std::string \~english Information string \~chinese 信息字符串
     */
    static std::string infoBandwidthCriterion(const BandwidthWeight* weight)
    {
        return std::string(GWM_LOG_TAG_BANDWIDTH_CIRTERION) + (weight->adaptive() ? "adaptive" : "fixed") + ",criterion";
    }

    /**
     * @brief \~english Get infomation of current bandwidth value and the corresponding criterion value.
     * \~chinese 获取当前带宽值和对应指标值的信息。
     * 
     * @param weight \~english Bandwidth weight \~chinese 带宽设置
     * @param value \~english Criterion value \~chinese 指标值
     * @return std::string \~english Information string \~chinese 信息字符串
     */
    static std::string infoBandwidthCriterion(const BandwidthWeight* weight, const double value)
    {
        if (weight->adaptive())
            return std::string(GWM_LOG_TAG_BANDWIDTH_CIRTERION) + std::to_string(int(weight->bandwidth())) + "," + std::to_string(value);
        else 
            return std::string(GWM_LOG_TAG_BANDWIDTH_CIRTERION) + std::to_string(weight->bandwidth()) + "," + std::to_string(value);
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