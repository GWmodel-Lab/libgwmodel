#include <string>
#include <mutex>
#include <gwmodel.h>
#include <queue>
#include <thread>
#include <condition_variable>
#include <fstream>

using namespace gwm;

class FileTelegram : public Logger
{
public:
    static std::queue<std::string> MessageQueue;

#ifdef ENABLE_OPENMP
    static std::mutex Lock;
    static bool TerminatePrinter;
#endif

    static void progress_print(std::ofstream& fout);

#ifdef ENABLE_OPENMP
    explicit FileTelegram(const std::string& filename) :
        mFileName(filename),
        mFileStream(filename.c_str())
    {
        mPrintThread = std::thread([this]() {
            progress_print(this->mFileStream);
        });
    }
#else
    explicit FileTelegram(const std::string& filename) : 
        mFileName(filename),
        mFileStream(filename.c_str())
    {}
#endif

    ~FileTelegram()
    {
#ifdef ENABLE_OPENMP
        TerminatePrinter = true;
        mPrintThread.join();
        mFileStream.close();
#endif
    }

    bool stop() override { return mCancelled; }

    void print(std::string message, LogLevel level, std::string fun_name, std::string file_name) override;

    void progress(std::size_t current, std::size_t total, std::string fun_name, std::string file_name) override;

    void progress(double percent, std::string fun_name, std::string file_name) override;

private:
    bool mCancelled = false;
    std::string mFileName = "log.txt";
    std::ofstream mFileStream;

#ifdef ENABLE_OPENMP
    std::thread mPrintThread;
    std::condition_variable mPrintable;
#endif

};
