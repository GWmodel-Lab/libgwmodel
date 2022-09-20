#ifndef CGWMSPATIALWEIGHT_H
#define CGWMSPATIALWEIGHT_H

#include <concepts>

#include "CGwmWeight.h"
#include "CGwmDistance.h"

#include "CGwmBandwidthWeight.h"

#include "CGwmCRSDistance.h"
#include "CGwmMinkwoskiDistance.h"
#include "CGwmDMatDistance.h"

/**
 * @brief A combined class of distance and weight. 
 * Instances of this class are usually constructed by providing pointers to CGwmDistance and CGwmWeight.
 * In the construct function, instances will be cloned.
 * This class provide the method CGwmSpatialWeight::weightVector() to calculate spatial weight directly.
 * 
 * If the distance and weight are set by pointers, this class will take the control of them, 
 * and when destructing the pointers will be deleted. 
 * If the distance and weight are set by references, this class will clone them. 
 */
template <class TW, class TD> requires std::derived_from<TW, CGwmWeight> && std::derived_from<TD, CGwmDistance>
class CGwmSpatialWeight
{
public:

    /**
     * @brief Construct a new CGwmSpatialWeight object.
     */
    CGwmSpatialWeight() {}

    /**
     * @brief Construct a new CGwmSpatialWeight object.
     * 
     * @param weight Pointer to a weight configuration.
     * @param distance Pointer to distance configuration.
     */
    CGwmSpatialWeight(const TW* weight, CGwmDistance* distance)
    {
        mWeight = weight;
        mDistance = distance;
    }

    /**
     * @brief Construct a new CGwmSpatialWeight object.
     * 
     * @param spatialWeight Reference to the object to copy from.
     */
    CGwmSpatialWeight(const CGwmSpatialWeight& spatialWeight)
    {
        mWeight = spatialWeight.mWeight;
        mDistance = spatialWeight.mDistance;
    }

    /**
     * @brief Destroy the CGwmSpatialWeight object.
     */
    virtual ~CGwmSpatialWeight() {}

    /**
     * @brief Get the pointer to CGwmSpatialWeight::mWeight .
     * 
     * @return Pointer to CGwmSpatialWeight::mWeight .
     */
    TW& weight() const
    {
        return static_cast<CGwmBandwidthWeight*>(mWeight);
    }

    /**
     * @brief Set the pointer to CGwmSpatialWeight::mWeight object.
     * 
     * @param weight Reference to CGwmWeight instance.
     * This object will be cloned.
     */
    void setWeight(const TW& weight)
    {
        mWeight = weight;
    }

    /**
     * @brief Set the pointer to CGwmSpatialWeight::mWeight object.
     * 
     * @param weight Reference to CGwmWeight instance.
     * This object will be cloned.
     */
    void setWeight(const TW&& weight)
    {
        mWeight = weight;
    }

    /**
     * @brief Get the pointer to CGwmSpatialWeight::mDistance.
     * 
     * @return Pointer to CGwmSpatialWeight::mDistance.
     */
    TD& distance() const
    {
        return mDistance;
    }

    /**
     * @brief Set the pointer to CGwmSpatialWeight::mDistance object.
     * 
     * @param distance Reference to CGwmDistance instance.
     * This object will be cloned.
     */
    void setDistance(const TD& distance)
    {
        mDistance = distance;
    }

    /**
     * @brief Set the pointer to CGwmSpatialWeight::mDistance object.
     * 
     * @param distance Reference to CGwmDistance instance.
     * This object will be cloned.
     */
    void setDistance(const TD&& distance)
    {
        mDistance = distance;
    }

public:

    /**
     * @brief Override operator = for this class. 
     * This function will first delete the current CGwmSpatialWeight::mWeight and CGwmSpatialWeight::mDistance,
     * and then clone CGwmWeight and CGwmDistance instances according pointers of the right value. 
     * 
     * @param spatialWeight Reference to the right value.
     * @return Reference of this object.
     */
    CGwmSpatialWeight& operator=(const CGwmSpatialWeight& spatialWeight)
    {
        if (this == &spatialWeight) return *this;
        mWeight = spatialWeight.mWeight;
        mDistance = spatialWeight.mDistance;
        return *this;
    }

    /**
     * @brief Override operator = for this class. 
     * This function will first delete the current CGwmSpatialWeight::mWeight and CGwmSpatialWeight::mDistance,
     * and then clone CGwmWeight and CGwmDistance instances according pointers of the right value. 
     * 
     * @param spatialWeight Right value reference to the right value.
     * @return Reference of this object.
     */
    CGwmSpatialWeight& operator=(const CGwmSpatialWeight&& spatialWeight)
    {
        if (this == &spatialWeight) return *this;
        mWeight = spatialWeight.mWeight;
        mDistance = spatialWeight.mDistance;
        return *this;
    }

public:

    /**
     * @brief 
     * 
     * @param parameter 
     * @param focus 
     * @return vec 
     */
    virtual vec weightVector(const DistanceParameter* parameter, uword focus)
    {
        return mWeight.weight(mDistance.distance(parameter, foucus));
    }

private:
    TW mWeight;      //!< Pointer to weight configuration.
    TD mDistance;  //!< Pointer to distance configuration.
};

#endif // CGWMSPATIALWEIGHT_H
