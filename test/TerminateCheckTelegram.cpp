#include "TerminateCheckTelegram.h"
#include <catch2/catch_all.hpp>

using namespace std;

queue<string> TerminateCheckTelegram::MessageQueue = queue<string>();
mutex TerminateCheckTelegram::Lock = mutex();
bool TerminateCheckTelegram::TerminatePrinter = false;

void TerminateCheckTelegram::progress_print()
{
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
    
}

void TerminateCheckTelegram::progress(size_t current, size_t total, string fun_name, string file_name)
{
    (void)current;
    (void)total;
    (void)file_name;
    unique_lock<mutex> locker(Lock);
    if (fun_name.find(mBreakStage, 0) >= 0)
    {
        mCancelled = true;
    }
    string ss = string("Progress") + fun_name + ": " + to_string(current) + "/" + to_string(total) + "\n";
    MessageQueue.push(ss);
}

void TerminateCheckTelegram::progress(double percent, string fun_name, string file_name)
{
    (void)percent;
    (void)file_name;
    unique_lock<mutex> locker(Lock);
    if (fun_name.find(mBreakStage, 0) >= 0)
    {
        mCancelled = true;
    }
    string ss = string("Progress") + fun_name + ": " + to_string(percent) + "%\n";
    MessageQueue.push(ss);
}
