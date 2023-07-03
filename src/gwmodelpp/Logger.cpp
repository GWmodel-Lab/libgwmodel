#include "Logger.h"

using namespace std;
using namespace gwm;

Logger::Printer Logger::printer = [](string message, Logger::LogLevel level, string fun_name, string file_name)
{
    (void)message;
    (void)level;
    (void)fun_name;
    (void)file_name;
};

Logger::Progresser Logger::progresser = [](std::size_t progress, std::size_t total)
{
    (void)progress;
    (void)total;
};

Logger::Stopper Logger::stopper = [](){
    return true;
};
