#include "GwmLogger.h"

using namespace std;

GwmLogger::Logger GwmLogger::logger = [](string message, GwmLogger::LogLevel level, string fun_name, string file_name)
{
    (void)message;
    (void)level;
    (void)fun_name;
    (void)file_name;
};