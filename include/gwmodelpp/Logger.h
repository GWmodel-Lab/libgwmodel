#ifndef GWMLOGGER_H
#define GWMLOGGER_H

#include <vector>
#include <string>
#include <numeric>

namespace gwm
{

/**
 * @brief \~english Join string vector with the specified delimeter.
 * \~chinese 使用指定的分隔符链接字符串向量。
 * 
 * @param delm \~english Delimeter \~chinese 分隔符
 * @param str_array \~english String vector to be joined \~chinese 要链接的字符串向量
 * @return std::string \~english Joined string \~chinese 链接的字符串
 */
inline std::string strjoin(const std::string& delm, const std::vector<std::string>& str_array)
{
    std::string ss = *str_array.cbegin();
    for (auto it = str_array.cbegin() + 1; it != str_array.cend(); it++)
        ss += (delm + *it);
    return ss;
}

/**
 * @brief \~english Interface for controller of algorithm. \~chinese 算法控制器接口
 * 
 */
struct ITelegram
{

    virtual ~ITelegram() {}

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

    /**
     * @brief \~english Call printer to print a log. \~chinese 调用打印函数输出日志
     * 
     * @param message \~english Log message \~chinese 日志消息
     * @param level \~english Log level \~chinese 日志等级
     * @param fun_name \~english Caller function name \~chinese 调用者名称
     * @param file_name \~english The file where caller function is defined \~chinese 调用者位于的文件
     */
    virtual void print(std::string message, ITelegram::LogLevel level, std::string fun_name, std::string file_name) = 0;

    /**
     * @brief \~english Report the progress of this algorithm. \~Chinese 报告该算法执行的进度。
     * 
     * @param current \~english Current progress. \~chinese 当前进度。
     * @param total \~english Total number of progress. \~chinese 进度刻度总数。
     */
    virtual void progress(std::size_t current, std::size_t total, std::string fun_name, std::string file_name) = 0;

    /**
     * @brief \~english Report the progress of this algorithm. \~Chinese 报告该算法执行的进度。
     * 
     * @param percent \~english Current percentage of total progress. \~chinese 当前进度相对于总进度的百分比。
     */
    virtual void progress(double percent, std::string fun_name, std::string file_name) = 0;

    /**
     * @brief \~english Tell the algorithm whether to stop. \~chinese 告诉算法是否要终止计算。
     * 
     * @return true \~english Yes, stop progress. \~chinese 是，停止计算。
     * @return false \~english No, don't stop progress. \~chinese 不，继续计算。
     */
    virtual bool stop() = 0;
};

/**
 * @brief \~english Logger. Used to pass logging messages to outer printer functions.
 * To accept messages, set the static member printer to self-defined functions.
 * \~chinese 日志记录器。用于向外部打印函数传递日志信息。
 * 如果要接受消息，将类中的静态成员变量 printer 设置为自定义的函数。
 * 
 */
class Logger : public ITelegram
{
public:

    /**
     * @brief \~english Construct a new Logger object. \~chinese 构造一个 Logger 对象。
     * 
     */
    Logger() {}

    /**
     * @brief \~english Destroy the Logger object. \~chinese 销毁一个 Logger 对象。
     * 
     */
    ~Logger() {}

    void print(std::string message, LogLevel level, std::string fun_name, std::string file_name) override
    {
        (void)message;
        (void)level;
        (void)fun_name;
        (void)file_name;
    }

    void progress(std::size_t current, std::size_t total, std::string fun_name, std::string file_name) override
    {
        (void)current;
        (void)total;
        (void)fun_name;
        (void)file_name;
    }

    void progress(double percent, std::string fun_name, std::string file_name) override
    {
        (void)percent;
        (void)fun_name;
        (void)file_name;
    }

    bool stop() override { return false; }
};

}



#endif  // GWMLOGGER_H