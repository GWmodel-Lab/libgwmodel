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
 * @brief Printer. Used to pass logging messages to outer printer functions.
 * To accept messages, set the static member printer to self-defined functions.
 */
class Logger
{
public:

    /**
     * @brief Level of logs.
     */
    enum class LogLevel
    {
        LOG_EMERG = 0, //!< The message says the system is unusable.
        LOG_ALERT = 1, //!< Action on the message must be taken immediately.
        LOG_CRIT = 2, //!< The message states a critical condition.
        LOG_ERR = 3, //!< The message describes an error.
        LOG_WARNING = 4, //!< The message is a warning.
        LOG_NOTICE = 5, //!< The message describes a normal but important event.
        LOG_INFO = 6, //!< The message is purely informational.
        LOG_DEBUG = 7 //!< The message is only for debugging purposes.
    };

    using Printer = std::function<void (std::string, LogLevel, std::string, std::string)>; //!< Printer type.

    static Printer printer;  //!< Printer used to print logging messages.

    /**
     * @brief Call printer to print a log;
     * 
     * @param message Log message.
     * @param level Log level.
     * @param fun_name Caller function name.
     * @param file_name The file where caller function is defined.
     */
    static void logging(std::string message, LogLevel level, std::string fun_name, std::string file_name)
    {
        printer(message, level, fun_name, file_name);
    }
};

}



#endif  // GWMLOGGER_H