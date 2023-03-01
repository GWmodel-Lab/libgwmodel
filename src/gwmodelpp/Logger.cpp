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