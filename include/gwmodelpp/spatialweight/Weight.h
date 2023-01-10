#ifndef WEIGHT_H
#define WEIGHT_H

#include <unordered_map>
#include <string>
#include <armadillo>


namespace gwm
{

/**
 * @brief Abstract base class for calculating weight from distance.
 * It defines some interface commonly used in spatial algorithms:
 * 
 * - Clone this object.
 * - Calculate distance vector for a focus point.
 * - Get maximum distance among all points.
 * - Get minimum distance among all points.
 * 
 * Pointer of this type can be put to these classes:
 * 
 * - BandwidthWeight
 * 
 */
class Weight
{
public:

    /**
     * @brief Type of weight.
     */
    enum WeightType
    {
        BandwidthWeight //!< Bandwidth weight
    };

    static std::unordered_map<WeightType, std::string> TypeNameMapper;

public:

    /**
     * @brief Construct a new Weight object.
     */
    Weight() {}

    /**
     * @brief Destroy the Weight object.
     */
    virtual ~Weight() {}

    /**
     * @brief Clone this object.
     * 
     * @return Newly created pointer.
     */
    virtual Weight* clone() = 0;

public:

    /**
     * @brief Calculate weight vector from a distance vector. 
     * 
     * @param dist According distance vector.
     * @return Weight vector.
     */
    virtual arma::vec weight(arma::vec dist) = 0;
};

}

#endif // WEIGHT_H
