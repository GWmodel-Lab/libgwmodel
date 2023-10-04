#include <string>
#include <mutex>
#include <gwmodel.h>
#include <queue>
#include <thread>
#include <condition_variable>
#include <fstream>

using namespace gwm;

class StdTelegram : public Logger
{
public:
    explicit StdTelegram()
    {}

    ~StdTelegram()
    {}

    bool stop() override { return false; }

    void print(std::string message, LogLevel level, std::string fun_name, std::string file_name) override;

    void progress(std::size_t current, std::size_t total, std::string fun_name, std::string file_name) override;

    void progress(double percent, std::string fun_name, std::string file_name) override;

};
