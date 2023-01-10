#ifndef GWMLOGGER_H
#define GWMLOGGER_H

#include <functional>
#include <string>

namespace gwm
{

#define GWM_LOGGING(MESSAGE, LEVEL) GwmLogger::logger((MESSAGE), (LEVEL), __FUNCTION__, __FILE__)

#define GWM_LOG_DEBUG(MESSAGE) GwmLogger::logger((MESSAGE), GwmLogger::LogLevel::LOG_DEBUG, __FUNCTION__, __FILE__)

#define GWM_LOG_INFO(MESSAGE) GwmLogger::logger((MESSAGE), GwmLogger::LogLevel::LOG_INFO, __FUNCTION__, __FILE__)

#define GWM_LOG_WARNNING(MESSAGE) GwmLogger::logger((MESSAGE), GwmLogger::LogLevel::LOG_WARNING, __FUNCTION__, __FILE__)

#define GWM_LOG_ERROR(MESSAGE) GwmLogger::logger((MESSAGE), GwmLogger::LogLevel::LOG_ERR, __FUNCTION__, __FILE__)

/**
 * @brief Logger. Used to pass logging messages to outer logger functions.
 * To accept messages, set the static member logger to self-defined functions.
 */
class GwmLogger
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

    using Logger = std::function<void (std::string, LogLevel, std::string, std::string)>; //!< Logger type.

    static Logger logger;  //!< Logger used to print logging messages.

    /**
     * @brief Call logger to print a log;
     * 
     * @param message Log message.
     * @param level Log level.
     * @param fun_name Caller function name.
     * @param file_name The file where caller function is defined.
     */
    static void logging(std::string message, LogLevel level, std::string fun_name, std::string file_name)
    {
        logger(message, level, fun_name, file_name);
    }
};

}



#endif  // GWMLOGGER_H