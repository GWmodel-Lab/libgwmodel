#ifndef IPARALLELIZABLE_H
#define IPARALLELIZABLE_H

namespace gwm
{

/**
 * @brief Define names of different parallel types.
 */
enum ParallelType
{
    SerialOnly = 1 << 0,    //!< Use no parallel methods.
    OpenMP = 1 << 1,        //!< Use multithread methods.
    CUDA = 1 << 2           //!< Use CUDA accelerated methods.
};

/**
 * @interface IParallelizable
 * @brief Interface for parallelizable algorithm. 
 * It defines some interface commonly used in parallelizable algorithms:
 * 
 * - Getter of parallel ability.
 * - Getter and setter of parallel type.
 * 
 * Pointer of this type can be put to these classes:
 * 
 * - GWRBasic
 * - GWSS
 * 
 */
struct IParallelizable
{
    /**
     * @brief Return the parallel ability of this algorithm.
     * 
     * @return Bitwise OR of aviliable parallel types of this algorithm.
     */
    virtual int parallelAbility() const = 0;

    /**
     * @brief Return the parallel type of this algorithm.
     * 
     * @return Parallel type of this algorithm 
     */
    virtual ParallelType parallelType() const = 0;

    /**
     * @brief Set the parallel type of this algorithm.
     * 
     * @param type Parallel type of this algorithm.
     */
    virtual void setParallelType(const ParallelType& type) = 0;
};

/**
 * @interface IOpenmpParallelizable
 * @brief Interface for parallelizable algorithm implemented by OpenMP. 
 * It defines some interface commonly used in parallelizable algorithms:
 * 
 * - Setter of thread numbers while paralleling.
 * 
 * Pointer of this type can be put to these classes:
 * 
 * - GWRBasic
 * - GWSS
 * 
 */
struct IOpenmpParallelizable : public IParallelizable
{
    /**
     * @brief Set the thread numbers while paralleling.
     * 
     * @param threadNum Number of threads.
     */
    virtual void setOmpThreadNum(const int threadNum) = 0;
};

/**
 * @interface ICudaParallelizable
 * @brief Interface for parallelizable algorithm implemented by CUDA. 
 * It defines some interface commonly used in parallelizable algorithms:
 * 
 * - Setter of ID of used GPU while paralleling. 
 * - Setter of group size while paralleling. 
 * 
 * Pointer of this type can be put to these classes:
 * 
 * - GWRBasic
 * - GWSS
 * 
 */
struct ICudaParallelizable : public IParallelizable
{
    /**
     * @brief Set ID of used GPU while paralleling. 
     * 
     * @param gpuId ID of used GPU. Start from 0.
     */
    virtual void setGPUId(const int gpuId) = 0;

    /**
     * @brief Set the group size while paralleling.
     * 
     * @param size Group size \f$g\f$ while paralleling. Usually this is up to memory size \f$m\f$ (B) of GPU with ID gpuId. For most GPU, 64 is OK.
     * If there are \f$n\f$ samples and \f$k\f$ variables, \f[ k * n * g * 8 < m \f]
     */
    virtual void setGroupSize(const double size) = 0;
};

}

#endif  // IPARALLELIZABLE_H