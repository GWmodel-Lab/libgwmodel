#ifndef BANDWIDTHSELECTOR_H
#define BANDWIDTHSELECTOR_H

#include <map>
#include <vector>
#include <utility>
#include "IBandwidthSelectable.h"
#include "spatialweight/BandwidthWeight.h"

namespace gwm
{

/**
 * @brief \~english Bandwidth selector based on golden-selection algorithm. \~chinese 基于黄金分割算法的带宽选择器
 * 
 */
class BandwidthSelector
{
public:

    /**
     * @brief \~english Construct a new Bandwidth Selector object. \~chinese 构造一个新的 BandwidthSelector 对象。
     */
    BandwidthSelector() {}

    /**
     * @brief \~english Construct a new Bandwidth Selector object. \~chinese 构造一个新的 BandwidthSelector 对象。
     * 
     * @param bandwidth \~english Bandwidth \~chinese 带宽
     * @param lower \~english Lower bound \~chinese 下限
     * @param upper \~english Upper bound \~chinese 上限
     */
    BandwidthSelector(BandwidthWeight* bandwidth, double lower, double upper) : mBandwidth(bandwidth) , mLower(lower) , mUpper(upper) {}

    /**
     * @brief \~english Destroy the Bandwidth Selector object. \~chinese 销毁 BandwidthSelector 对象。
     */
    ~BandwidthSelector() {}

public:

    /**
     * @brief \~english Get the bandwidth. \~chinese 获取带宽。
     * 
     * @return BandwidthWeight* \~english Bandwidth \~chinese 带宽
     */
    BandwidthWeight *bandwidth() const { return mBandwidth; }

    /**
     * @brief \~english Set the bandwidth. \~chinese 设置带宽。
     * 
     * @param bandwidth \~english Bandwidth \~chinese 带宽
     */
    void setBandwidth(BandwidthWeight *bandwidth) { mBandwidth = bandwidth; }

    /**
     * @brief \~english Get the lower bound. \~chinese 获取下限。
     * 
     * @return double \~english Lower bound \~chinese 下限
     */
    double lower() const { return mLower; }

    /**
     * @brief \~english Set the lower bound. \~chinese 设置下限。
     * 
     * @param lower \~english Lower bound \~chinese 下限
     */
    void setLower(double lower) { mLower = lower; }

    /**
     * @brief \~english Get the upper bound. \~chinese 获取上限。
     * 
     * @return double \~english Upper bound \~chinese 上限
     */
    double upper() const { return mUpper; }

    /**
     * @brief \~english Set the upper bound. \~chinese 设置上限。
     * 
     * @param upper \~english Upper bound \~chinese 上限
     */
    void setUpper(double upper) { mUpper = upper; }

    /**
     * @brief \~english Get the list of criterion values for each bandwidth. \~chinese 获取带宽优选过程中每种带宽对应的指标值列表。
     * 
     * @return VariablesCriterionList \~english List of criterion values for each bandwidth \~chinese 带宽优选过程中每种带宽对应的指标值列表
     */
    BandwidthCriterionList bandwidthCriterion() const;

public:

    /**
     * @brief \~english Optimize bandwidth. \~chinese 优化带宽。
     * 
     * @param instance \~english A pointer to a instance of type inherited from gwm::IBandwidthSelectable \~chinese 指向派生自 gwm::IBandwidthSelectable 类型对象的指针
     * @return std::vector<std::size_t> \~english Optimized bandwdith \~chinese 优选后的带宽
     */
    BandwidthWeight* optimize(IBandwidthSelectable* instance);

private:
    BandwidthWeight* mBandwidth;    //!< \~english Bandwidth \~chinese 带宽
    double mLower;  //!< \~english Lower bound \~chinese 下限
    double mUpper;  //!< \~english Upper bound \~chinese 上限
    std::unordered_map<double, double> mBandwidthCriterion; //!< \~english List of criterion values for each bandwidth \~chinese 带宽优选过程中每种带宽对应的指标值列表
};

}

#endif  // BANDWIDTHSELECTOR_H