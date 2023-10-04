#include "StdTelegram.h"
#include <iostream>

using namespace std;

void StdTelegram::print(string message, gwm::Logger::LogLevel level, string fun_name, string file_name)
{
    cout << message + "\n";
}

void StdTelegram::progress(size_t current, size_t total, string fun_name, string file_name)
{
    (void)current;
    (void)total;
    (void)fun_name;
    (void)file_name;
}

void StdTelegram::progress(double percent, string fun_name, string file_name)
{
    (void)percent;
    (void)file_name;
    (void)fun_name;
}
