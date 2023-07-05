#include "TerminateCheckTelegram.h"
#include <catch2/catch_all.hpp>

std::queue<std::string> TerminateCheckTelegram::MessageQueue = std::queue<std::string>();
std::mutex TerminateCheckTelegram::Lock = std::mutex();
bool TerminateCheckTelegram::TerminatePrinter = false;

void TerminateCheckTelegram::progress_print()
{
    while (!TerminatePrinter)
    {
        std::unique_lock<std::mutex> locker(Lock);
        if (MessageQueue.size() > 0)
        {
            auto msg = MessageQueue.front();
            MessageQueue.pop();
            INFO(msg);
        }
    }
    
}

void TerminateCheckTelegram::progress(std::size_t current, std::size_t total, std::string fun_name, std::string file_name)
{
    (void)current;
    (void)total;
    (void)file_name;
    std::unique_lock<std::mutex> locker(Lock);
    if (fun_name.rfind(mBreakStage, 0) == 0)
    {
        mCancelled = true;
    }
    std::stringstream ss;
    ss << "Progress" << fun_name << ": " << current << "/" << total << "\n";
    MessageQueue.push(ss.str());
}

void TerminateCheckTelegram::progress(double percent, std::string fun_name, std::string file_name)
{
    (void)percent;
    (void)file_name;
    std::unique_lock<std::mutex> locker(Lock);
    if (fun_name.rfind(mBreakStage, 0) == 0)
    {
        mCancelled = true;
    }
    std::stringstream ss;
    ss << "Progress" << fun_name << ": " << percent << "%\n";
    MessageQueue.push(ss.str());
}
