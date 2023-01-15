#ifndef SPATIALWEIGHT_H
#define SPATIALWEIGHT_H

#include "Weight.h"
#include "Distance.h"

#include "BandwidthWeight.h"

#include "CRSDistance.h"
#include "MinkwoskiDistance.h"
#include "DMatDistance.h"
#include "OneDimDistance.h"

namespace gwm
{

/**
 * @brief A combined class of distance and weight. 
 * Instances of this class are usually constructed by providing pointers to Distance and Weight.
 * In the construct function, instances will be cloned.
 * This class provide the method SpatialWeight::weightVector() to calculate spatial weight directly.
 * 
 * If the distance and weight are set by pointers, this class will take the control of them, 
 * and when destructing the pointers will be deleted. 
 * If the distance and weight are set by references, this class will clone them. 
 */
class SpatialWeight
{
public:

    /**
     * @brief Construct a new SpatialWeight object.
     */
    SpatialWeight();

    /**
     * @brief Construct a new SpatialWeight object.
     * 
     * @param weight Pointer to a weight configuration.
     * @param distance Pointer to distance configuration.
     */
    SpatialWeight(Weight* weight, Distance* distance);

    /**
     * @brief Construct a new SpatialWeight object.
     * 
     * @param spatialWeight Reference to the object to copy from.
     */
    SpatialWeight(const SpatialWeight& spatialWeight);

    /**
     * @brief Destroy the SpatialWeight object.
     */
    virtual ~SpatialWeight();

    /**
     * @brief Get the pointer to SpatialWeight::mWeight .
     * 
     * @return Pointer to SpatialWeight::mWeight .
     */
    Weight *weight() const
    {
        return mWeight;
    }

    /**
     * @brief Set the pointer to SpatialWeight::mWeight object.
     * 
     * @param weight Pointer to Weight instance. 
     * Control of this pointer will be taken, and it will be deleted when destructing.
     */
    void setWeight(Weight *weight)
    {
        if (mWeight) delete mWeight;
        mWeight = weight;
    }

    /**
     * @brief Set the pointer to SpatialWeight::mWeight object.
     * 
     * @param weight Reference to Weight instance.
     * This object will be cloned.
     */
    void setWeight(Weight& weight)
    {
        if (mWeight) delete mWeight;
        mWeight = weight.clone();
    }

    /**
     * @brief Set the pointer to SpatialWeight::mWeight object.
     * 
     * @param weight Reference to Weight instance.
     * This object will be cloned.
     */
    void setWeight(Weight&& weight)
    {
        if (mWeight) delete mWeight;
        mWeight = weight.clone();
    }

    /**
     * @brief Get the pointer to SpatialWeight::mWeight and cast it to required type.
     * 
     * @tparam T Type of return value. Only BandwidthWeight is allowed.
     * @return Casted pointer to SpatialWeight::mWeight.
     */
    template<typename T>
    T* weight() const { return nullptr; }

    /**
     * @brief Get the pointer to SpatialWeight::mDistance.
     * 
     * @return Pointer to SpatialWeight::mDistance.
     */
    Distance *distance() const
    {
        return mDistance;
    }

    /**
     * @brief Set the pointer to SpatialWeight::mDistance object.
     * 
     * @param distance Pointer to Distance instance. 
     * Control of this pointer will be taken, and it will be deleted when destructing.
     */
    void setDistance(Distance *distance)
    {
        if (mDistance) delete mDistance;
        mDistance = distance;
    }

    /**
     * @brief Set the pointer to SpatialWeight::mDistance object.
     * 
     * @param distance Reference to Distance instance.
     * This object will be cloned.
     */
    void setDistance(Distance& distance)
    {
        if (mDistance) delete mDistance;
        mDistance = distance.clone();
    }

    /**
     * @brief Set the pointer to SpatialWeight::mDistance object.
     * 
     * @param distance Reference to Distance instance.
     * This object will be cloned.
     */
    void setDistance(Distance&& distance)
    {
        if (mDistance) delete mDistance;
        mDistance = distance.clone();
    }

    /**
     * @brief Get the pointer to SpatialWeight::mDistance and cast it to required type.
     * 
     * @tparam T Type of return value. Only CRSDistance and MinkwoskiDistance is allowed.
     * @return Casted pointer to SpatialWeight::mDistance.
     */
    template<typename T>
    T* distance() const { return nullptr; }

public:

    /**
     * @brief Override operator = for this class. 
     * This function will first delete the current SpatialWeight::mWeight and SpatialWeight::mDistance,
     * and then clone Weight and Distance instances according pointers of the right value. 
     * 
     * @param spatialWeight Reference to the right value.
     * @return Reference of this object.
     */
    SpatialWeight& operator=(const SpatialWeight& spatialWeight);

    /**
     * @brief Override operator = for this class. 
     * This function will first delete the current SpatialWeight::mWeight and SpatialWeight::mDistance,
     * and then clone Weight and Distance instances according pointers of the right value. 
     * 
     * @param spatialWeight Right value reference to the right value.
     * @return Reference of this object.
     */
    SpatialWeight& operator=(const SpatialWeight&& spatialWeight);

public:

    /**
     * @brief 
     * 
     * @param parameter 
     * @param focus 
     * @return arma::vec 
     */
    virtual arma::vec weightVector(arma::uword focus)
    {
        return mWeight->weight(mDistance->distance(focus));
    }

    /**
     * @brief Get whether this object is valid in geting weight vector.
     * 
     * @return true if this object is valid.
     * @return false if this object is invalid.
     */
    virtual bool isValid();

private:
    Weight* mWeight = nullptr;      //!< Pointer to weight configuration.
    Distance* mDistance = nullptr;  //!< Pointer to distance configuration.
};

template<>
inline BandwidthWeight* SpatialWeight::weight<BandwidthWeight>() const
{
    return static_cast<BandwidthWeight*>(mWeight);
}

template<>
inline CRSDistance* SpatialWeight::distance<CRSDistance>() const
{
    return static_cast<CRSDistance*>(mDistance);
}

template<>
inline MinkwoskiDistance* SpatialWeight::distance<MinkwoskiDistance>() const
{
    return static_cast<MinkwoskiDistance*>(mDistance);
}

template<>
inline DMatDistance* SpatialWeight::distance<DMatDistance>() const
{
    return static_cast<DMatDistance*>(mDistance);
}

template<>
inline OneDimDistance* SpatialWeight::distance<OneDimDistance>() const
{
    return static_cast<OneDimDistance*>(mDistance);
}

}

#endif // SPATIALWEIGHT_H
