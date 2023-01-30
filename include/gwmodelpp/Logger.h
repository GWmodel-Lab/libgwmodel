#ifndef GWMLOGGER_H
#define GWMLOGGER_H

#include <functional>
#include <string>

#define GWM_LOGGING(MESSAGE, LEVEL) gwm::Logger::printer((MESSAGE), (LEVEL), __FUNCTION__, __FILE__)

#define GWM_LOG_DEBUG(MESSAGE) gwm::Logger::printer((MESSAGE), Logger::LogLevel::LOG_DEBUG, __FUNCTION__, __FILE__)

#define GWM_LOG_INFO(MESSAGE) gwm::Logger::printer((MESSAGE), Logger::LogLevel::LOG_INFO, __FUNCTION__, __FILE__)

#define GWM_LOG_WARNNING(MESSAGE) gwm::Logger::printer((MESSAGE), Logger::LogLevel::LOG_WARNING, __FUNCTION__, __FILE__)

#define GWM_LOG_ERROR(MESSAGE) gwm::Logger::printer((MESSAGE), Logger::LogLevel::LOG_ERR, __FUNCTION__, __FILE__)

namespace gwm
{

/**
 * @brief \~english Logger. Used to pass logging messages to outer printer functions.
 * To accept messages, set the static member printer to self-defined functions.
 * \~chinese 日志记录器。用于向外部打印函数传递日志信息。
 * 如果要接受消息，将类中的静态成员变量 printer 设置为自定义的函数。
 * 
 */
class Logger
{
public:

    /**
     * @brief \~english Level of logs. \~chinese 日志等级。
     */
    enum class LogLevel
    {
        LOG_EMERG = 0, //!< \~english The message says the system is unusable \~chinese 系统完全不可用
        LOG_ALERT = 1, //!< \~english Action on the message must be taken immediately \~chinese 需要执行操作
        LOG_CRIT = 2, //!< \~english The message states a critical condition \~chinese 严重情况
        LOG_ERR = 3, //!< \~english The message describes an error \~chinese 错误
        LOG_WARNING = 4, //!< \~english The message is a warning \~chinese 警告
        LOG_NOTICE = 5, //!< \~english The message describes a normal but important event \~chinese 注意
        LOG_INFO = 6, //!< \~english The message is purely informational \~chinese 信息
        LOG_DEBUG = 7 //!< \~english The message is only for debugging purposes \~chinese 调试
    };

    using Printer = std::function<void (std::string, LogLevel, std::string, std::string)>; //!< Printer type.

    static Printer printer;  //!< Printer used to print logging messages.

    /**
     * @brief \~english Call printer to print a log. \~chinese 调用打印函数输出日志
     * 
     * @param message \~english Log message \~chinese 日志消息
     * @param level \~english Log level \~chinese 日志等级
     * @param fun_name \~english Caller function name \~chinese 调用者名称
     * @param file_name \~english The file where caller function is defined \~chinese 调用者位于的文件
     */
    static void logging(std::string message, LogLevel level, std::string fun_name, std::string file_name)
    {
        printer(message, level, fun_name, file_name);
    }
};

}



#endif  // GWMLOGGER_H