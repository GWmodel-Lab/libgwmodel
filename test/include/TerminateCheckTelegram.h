#include <string>
#include <mutex>
#include <gwmodel.h>
#include <queue>
#include <thread>
#include <condition_variable>

using namespace gwm;

class TerminateCheckTelegram : public Logger
{
public:

    static std::queue<std::string> MessageQueue;
    
    static std::mutex Lock;

    static void progress_print();

    static bool TerminatePrinter;

    TerminateCheckTelegram(std::string breakStage, std::size_t breakProgress) : 
        mBreakStage(breakStage),
        mBreakProgress(breakProgress),
        mPrintThread(progress_print)
    {}

    ~TerminateCheckTelegram()
    {
        TerminatePrinter = true;
        mPrintThread.join();
    }

    bool stop() override { return mCancelled; }

    void progress(std::size_t current, std::size_t total, std::string fun_name, std::string file_name) override;

    void progress(double percent, std::string fun_name, std::string file_name) override;

private:
    bool mCancelled = false;
    std::string mBreakStage = "";
    std::size_t mBreakProgress = 0;
    std::thread mPrintThread;
    std::condition_variable mPrintable;
};

const std::map<ParallelType, std::string> ParallelTypeDict = {
    std::make_pair(ParallelType::SerialOnly, "SerialOnly"),
    std::make_pair(ParallelType::OpenMP, "OpenMP"),
    std::make_pair(ParallelType::CUDA, "CUDA")
};
