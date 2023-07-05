#include <string>
#include <gwmodel.h>

using namespace gwm;

class TerminateCheckTelegram : public Logger
{
public:
    TerminateCheckTelegram(std::string breakStage, std::size_t breakProgress) : mBreakStage(breakStage), mBreakProgress(breakProgress) {}

    bool stop() override { return mCancelled; }

    void progress(std::size_t current, std::size_t total, std::string fun_name, std::string file_name) override
    {
        (void)current;
        (void)total;
        (void)file_name;
        if (fun_name.rfind(mBreakStage, 0) == 0)
        {
            mCancelled = true;
        }
        INFO("Progress" << fun_name << ": " << current << "/" << total << "\n");
    }

    void progress(double percent, std::string fun_name, std::string file_name) override
    {
        (void)percent;
        (void)file_name;
        if (fun_name.rfind(mBreakStage, 0) == 0)
        {
            mCancelled = true;
        }
        INFO("Progress" << fun_name << ": " << percent << "%\n");
    }

private:
    bool mCancelled = false;
    std::string mBreakStage = "";
    std::size_t mBreakProgress = 0;
};