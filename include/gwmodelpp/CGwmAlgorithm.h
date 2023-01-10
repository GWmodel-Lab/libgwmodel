#ifndef CGWMALGORITHM_H
#define CGWMALGORITHM_H

namespace gwm
{

/**
 * @brief Abstract algorithm class.
 * This class cannot been constructed.
 * It defines some interface commonly used in spatial algorithms:
 * 
 * - Run the algorithm.
 * 
 * Pointer of this type can be put to these classes:
 * 
 * - CGwmGWRBasic
 * - CGwmGWSS
 * 
 */
class CGwmAlgorithm
{
public:

    /**
     * @brief Construct a new CGwmAlgorithm object.
     */
    CGwmAlgorithm() {}

    /**
     * @brief Destroy the CGwmAlgorithm object.
     */
    virtual ~CGwmAlgorithm() {}

public:

    /**
     * @brief Check whether the algorithm's configuration is valid. 
     * 
     * @return true if the algorithm's configuration is valid.
     * @return false if the algorithm's configuration is invalid.
     */
    virtual bool isValid() = 0;
};

}

#endif  // CGWMALGORITHM_H