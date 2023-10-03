#include "TerminateCheckTelegram.h"
#include <catch2/catch_all.hpp>

using namespace std;

queue<string> TerminateCheckTelegram::MessageQueue = queue<string>();
#ifdef ENABLE_OPENMP
mutex TerminateCheckTelegram::Lock = mutex();
bool TerminateCheckTelegram::TerminatePrinter = false;
#endif

void TerminateCheckTelegram::progress_print()
{
#ifdef ENABLE_OPENMP
   while (!TerminatePrinter)
    {
        unique_lock<mutex> locker(Lock);
        if (MessageQueue.size() > 0)
        {
            auto msg = MessageQueue.front();
            MessageQueue.pop();
            INFO(msg);
        }
    }
#else
    INFO(MessageQueue.front());
    MessageQueue.pop();
#endif
}

void TerminateCheckTelegram::progress(size_t current, size_t total, string fun_name, string file_name)
{
    (void)current;
    (void)total;
    (void)file_name;
#ifdef ENABLE_OPENMP
    unique_lock<mutex> locker(Lock);
#endif
    if (fun_name.find(mBreakStage, 0) >= 0 && current >= mBreakProgress)
    {
        mCancelled = true;
    }
    string ss = string("Progress") + fun_name + ": " + to_string(current) + "/" + to_string(total) + "\n";
    MessageQueue.push(ss);
#ifndef ENABLE_OPENMP
    progress_print();
#endif
}

void TerminateCheckTelegram::progress(double percent, string fun_name, string file_name)
{
    (void)percent;
    (void)file_name;
#ifdef ENABLE_OPENMP
    unique_lock<mutex> locker(Lock);
#endif
    if (fun_name.find(mBreakStage, 0) >= 0 && (percent * 100) >= mBreakProgress)
    {
        mCancelled = true;
    }
    string ss = string("Progress") + fun_name + ": " + to_string(percent) + "%\n";
    MessageQueue.push(ss);
#ifndef ENABLE_OPENMP
    progress_print();
#endif
}
