#ifndef ALGORITHM_H
#define ALGORITHM_H

#include <cfloat>
#include <memory>
#include <sstream>
#include "Status.h"
#include "Logger.h"

namespace gwm
{

/**
 * @brief 
 * \~english Shortcut to print log with function name and file name. 
 * \~chinese 用于输出带有函数名和文件名的日志的快捷宏函数。
 * 
 * @param MESSAGE \~english Message \~chinese 日志消息
 * @param LEVEL \~english Level of this message \~chinese 消息等级
 */
#define GWM_LOGGING(MESSAGE, LEVEL) this->mTelegram->print((MESSAGE), (LEVEL), __FUNCTION__, __FILE__);

/**
 * @brief 
 * \~english Shortcut to print debug log with function name and file name. 
 * \~chinese 用于输出带有函数名和文件名的调试日志的快捷宏函数。
 * 
 * @param MESSAGE \~english Message \~chinese 日志消息
 */
#define GWM_LOG_DEBUG(MESSAGE) this->mTelegram->print((MESSAGE), Logger::LogLevel::LOG_DEBUG, __FUNCTION__, __FILE__);

/**
 * @brief 
 * \~english Shortcut to print info log with function name and file name. 
 * \~chinese 用于输出带有函数名和文件名的消息日志的快捷宏函数。
 * 
 * @param MESSAGE \~english Message \~chinese 日志消息
 */
#define GWM_LOG_INFO(MESSAGE) this->mTelegram->print((MESSAGE), Logger::LogLevel::LOG_INFO, __FUNCTION__, __FILE__);

/**
 * @brief 
 * \~english Shortcut to print warning log with function name and file name. 
 * \~chinese 用于输出带有函数名和文件名的警告日志的快捷宏函数。
 * 
 * @param MESSAGE \~english Message \~chinese 日志消息
 */
#define GWM_LOG_WARNNING(MESSAGE) this->mTelegram->print((MESSAGE), Logger::LogLevel::LOG_WARNING, __FUNCTION__, __FILE__);

/**
 * @brief 
 * \~english Shortcut to print error log with function name and file name. 
 * \~chinese 用于输出带有函数名和文件名的错误日志的快捷宏函数。
 * 
 * @param MESSAGE \~english Message \~chinese 日志消息
 */
#define GWM_LOG_ERROR(MESSAGE) this->mTelegram->print((MESSAGE), Logger::LogLevel::LOG_ERR, __FUNCTION__, __FILE__);

/**
 * @brief 
 * \~english Check whether to stop. If yes, set the `STATUS` to `gwm::Status::Terminated` and call `break` to stop.
 * \~chinese 检查是否需要停止。如果要，将 `STATUS` 设置为 `gwm::Status::Terminated` 并使用 `break` 停止。
 * 
 * @param STATUS \~english Variable to store status value \~chinese 保存状态值的变量
 */
#define GWM_LOG_STOP_BREAK(STATUS) { if (this->mTelegram->stop()) { STATUS = gwm::Status::Terminated; break;} };

/**
 * @brief 
 * \~english Check whether to stop. If yes, set the `STATUS` to `gwm::Status::Terminated` and call `continue` to skip loop.
 * \~chinese 检查是否需要停止。如果要，将 `STATUS` 设置为 `gwm::Status::Terminated` 并使用 `continue` 跳过循环。
 * 
 * @param STATUS \~english Variable to store status value \~chinese 保存状态值的变量
 */
#define GWM_LOG_STOP_CONTINUE(STATUS) { if (this->mTelegram->stop()) { STATUS = gwm::Status::Terminated; continue;} };

/**
 * @brief 
 * \~english Check whether to stop. If yes, set the `STATUS` to `gwm::Status::Terminated` and call `return` to stop.
 * \~chinese 检查是否需要停止。如果要，将 `STATUS` 设置为 `gwm::Status::Terminated` 并使用 `return` 停止。
 * 
 * @param STATUS \~english Variable to store status value \~chinese 保存状态值的变量
 * @param REVAL \~english Value to return \~chinese 返回值
 */
#define GWM_LOG_STOP_RETURN(STATUS, REVAL) { if (this->mTelegram->stop()) { STATUS = gwm::Status::Terminated; return (REVAL);} }

/**
 * @brief Shortcut to report progress of current and total numbers with function name and file name. 
 * 
 * @param CURRENT \~english Current progress \~chinese 当前进度值
 * @param TOTAL \~english Progress total value \~chinese 总进度值
 */
#define GWM_LOG_PROGRESS(CURRENT, TOTAL) { this->mTelegram->progress((CURRENT), (TOTAL), (__FUNCTION__), (__FILE__)); };

/**
 * @brief Shortcut to report progress of percentage numbers with function name and file name. 
 * 
 * @param PERCENT \~english Current percentage of progress \~chinese 当前进度的百分比
 */
#define GWM_LOG_PROGRESS_PERCENT(PERCENT) { this->mTelegram->progress((PERCENT), (__FUNCTION__), (__FILE__)); };

#define GWM_LOG_TAG_STAGE "#stage "

/**
 * @brief Shortcut to report stages of an algorithm.
 * 
 * @param STAGE \~english Stage description \~chinese 当前阶段的描述
 */
#define GWM_LOG_STAGE(STAGE) { GWM_LOG_INFO((std::stringstream() << (GWM_LOG_TAG_STAGE) << (STAGE)).str()); }

/**
 * \~english
 * @brief Abstract algorithm class.
 * This class cannot been constructed.
 * It defines some interface commonly used in spatial algorithms
 * 
 * \~chinese
 * @brief 抽象算法基类。
 * 该类无法被构造。该类型定义了一些在空间算法中常用的接口。
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
    Algorithm() : mTelegram(new Logger()) {}

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
     * @brief \~english Ge the reference of pointer to Telegram object. \~chinese 返回 Telegram 指针的引用。
     * 
     * @return const std::unique_ptr<ITelegram>& \~english Reference of pointer to Telegram object \~chinese Telegram 指针的引用
     */
    const std::unique_ptr<ITelegram>& telegram() const
    {
        return mTelegram;
    }

    /**
     * @brief \~english Set the Telegram pointer. \~chinese 设置 Telegram 指针。
     * 
     * @param telegram \~english The pointer to the new Telegram object. This instance will take the management of pointer `telegram`.
     * \~chinese 新的 Telegram 对象的指针。所有权将被该 Algorithm 实例接管。
     */
    void setTelegram(std::unique_ptr<ITelegram> telegram)
    {
        mTelegram = std::move(telegram);
    }

    /**
     * @brief Manually send a debug message via telegram.
     * 
     * @param message Telegram message.
     * @param function Caller's name.
     * @param file Name of the file where caller defined.
     */
    void debug(std::string message, std::string function, std::string file)
    { 
        mTelegram->print(message, ITelegram::LogLevel::LOG_DEBUG, function, file);
    }

    /**
     * @brief Get the status of this algorithm.
     * 
     * @return const Status of this algorithm.
     */
    const Status status() const { return mStatus; }

public:

    /**
     * \~english
     * @brief Check whether the algorithm's configuration is valid. 
     * 
     * @return true if the algorithm's configuration is valid.
     * @return false if the algorithm's configuration is invalid.
     * 
     * \~chinese
     * @brief 检查算法配置是否合法。 
     * 
     * @return true 如果算法配置是合法的。
     * @return false 如果算法配置不合法。
     */
    virtual bool isValid() = 0;

protected:

    /**
     * @brief Set the Status of this algorithm.
     * 
     * @param status Status of this algorithm
     */
    void setStatus(Status status) { mStatus = status; }

protected:
    std::unique_ptr<ITelegram> mTelegram = nullptr; //!< \~english Pointer to the `ITelegram` instance \~chinese 指向 `ITelegram` 实例的指针
    Status mStatus = Status::Success; //!< \~english Algorithm status \~chinese 算法状态
};

}

#endif  // ALGORITHM_H