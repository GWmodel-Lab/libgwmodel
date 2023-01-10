#ifndef ALGORITHM_H
#define ALGORITHM_H

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
 * - GWRBasic
 * - GWSS
 * 
 */
class Algorithm
{
public:

    /**
     * @brief Construct a new Algorithm object.
     */
    Algorithm() {}

    /**
     * @brief Destroy the Algorithm object.
     */
    virtual ~Algorithm() {}

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

#endif  // ALGORITHM_H