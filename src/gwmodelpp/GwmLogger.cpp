#include "GwmLogger.h"

using namespace std;

GwmLogger::Logger GwmLogger::logger = [](string message, GwmLogger::LogLevel level, string fun_name, string file_name)
{
    (message);
    (level);
    (fun_name);
    (file_name);
};