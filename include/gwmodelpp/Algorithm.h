#ifndef ALGORITHM_H
#define ALGORITHM_H

namespace gwm
{

/**
 * \~english
 * @brief Abstract algorithm class.
 * This class cannot been constructed.
 * It defines some interface commonly used in spatial algorithms:
 * 
 *  - Run the algorithm.
 *  - Check the configuretion.
 * 
 * \~chinese
 * @brief 抽象算法基类。
 * 该类无法被构造。该类型定义了一些在空间算法中常用的接口，如：
 * 
 *  - 运行算法。
 *  - 检查算法配置是否合法。
 * 
 */
class Algorithm
{
public:

    /**
     * \~english 
     * @brief Construct a new Algorithm object.
     * 
     * \~chinese
     * @brief 构造一个新的 Algorithm 类对象。
     * 
     */
    Algorithm() {}

    /**
     * \~english 
     * @brief Destroy the Algorithm object.
     * 
     * \~chinese
     * @brief 销毁 Algorithm 类对象。
     */
    virtual ~Algorithm() {}

public:

    /**
     * @brief Check whether the algorithm's configuration is valid. 
     * 
     * @return true if the algorithm's configuration is valid.
     * @return false if the algorithm's configuration is invalid.
     * 
     * \~chinese
     * @brief 检查算法配置是否合法。 
     * 
     * @return true 如果算法配置合法。
     * @return false 如果算法配置不合法。
     */
    virtual bool isValid() = 0;
};

}

#endif  // ALGORITHM_H