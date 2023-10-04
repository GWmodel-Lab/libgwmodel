#include "FileTelegram.h"
#include <catch2/catch_all.hpp>

using namespace std;

queue<string> FileTelegram::MessageQueue = queue<string>();
#ifdef ENABLE_OPENMP
mutex FileTelegram::Lock = mutex();
bool FileTelegram::TerminatePrinter = false;
#endif

void FileTelegram::progress_print(ofstream& fout)
{
#ifdef ENABLE_OPENMP
   while (!TerminatePrinter)
    {
        unique_lock<mutex> locker(Lock);
        if (MessageQueue.size() > 0)
        {
            auto msg = MessageQueue.front();
            MessageQueue.pop();
            fout << msg;
        }
    }
#else
    INFO(MessageQueue.front());
    MessageQueue.pop();
#endif
}

void FileTelegram::print(string message, gwm::Logger::LogLevel level, string fun_name, string file_name)
{
#ifdef ENABLE_OPENMP
    unique_lock<mutex> locker(Lock);
#endif
    string ss = message + "\n";
    MessageQueue.push(ss);
#ifndef ENABLE_OPENMP
    progress_print(mFileStream);
#endif
}

void FileTelegram::progress(size_t current, size_t total, string fun_name, string file_name)
{
    (void)current;
    (void)total;
    (void)fun_name;
    (void)file_name;
}

void FileTelegram::progress(double percent, string fun_name, string file_name)
{
    (void)percent;
    (void)file_name;
    (void)fun_name;
}
