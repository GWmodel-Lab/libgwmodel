#define CATCH_CONFIG_MAIN

#include <catch2/catch_all.hpp>

#include <vector>
#include <string>
#include <armadillo>
#include "gwmodelpp/GWRMultiscale.h"
#include "gwmodelpp/GWSS.h"

#include "gwmodelpp/spatialweight/CRSDistance.h"
#include "gwmodelpp/spatialweight/BandwidthWeight.h"
#include "gwmodelpp/spatialweight/SpatialWeight.h"
#include "gwmodelpp/Logger.h"
#include "londonhp100.h"

using namespace std;
using namespace arma;
using namespace gwm;

struct MyLoggerInspector
{
    bool called = false;
    bool progressed = false;
};

struct MyLogger : Logger
{
    MyLogger(MyLoggerInspector* pInspector): inspector(pInspector) {}

    ~MyLogger() {}

    void print(string message, ITelegram::LogLevel level, string fun_name, string file_name) override
    {
        cout << "[" << fun_name << "] (in file " << file_name << "): " << message << "\n";
        inspector->called = true;
    }

    void progress(size_t current, size_t total, string fun_name, string file_name) override
    {
        cout << "[progress]:" << current << "/" << total << "(in file " << file_name << "): " << fun_name << "\n";
        inspector->progressed = true;
    }

    MyLoggerInspector* inspector;
};

TEST_CASE("Logging")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_data.n_rows), londonhp100_data.cols(1, 3));

    SECTION("printer")
    {
        MyLoggerInspector inspector;
        auto logger = make_unique<MyLogger>(&inspector);

        uword nVar = 3;
        vector<SpatialWeight> spatials;
        vector<bool> preditorCentered;
        vector<GWRMultiscale::BandwidthInitilizeType> bandwidthInitialize;
        vector<GWRMultiscale::BandwidthSelectionCriterionType> bandwidthSelectionApproach;
        for (size_t i = 0; i < nVar; i++)
        {
            CRSDistance distance;
            BandwidthWeight bandwidth(0, false, BandwidthWeight::Bisquare);
            spatials.push_back(SpatialWeight(&bandwidth, &distance));
            preditorCentered.push_back(i != 0);
            bandwidthInitialize.push_back(GWRMultiscale::BandwidthInitilizeType::Null);
            bandwidthSelectionApproach.push_back(GWRMultiscale::BandwidthSelectionCriterionType::CV);
        }
        bandwidthInitialize.pop_back();

        GWRMultiscale algorithm;
        algorithm.setTelegram(std::move(logger));
        algorithm.setCoords(londonhp100_coord);
        algorithm.setDependentVariable(y);
        algorithm.setIndependentVariables(x);
        algorithm.setSpatialWeights(spatials);
        algorithm.setHasHatMatrix(true);
        try
        {
            algorithm.setBandwidthInitilize(bandwidthInitialize);
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }
        

        REQUIRE(inspector.called);
    }

    SECTION("progress")
    {
        MyLoggerInspector inspector;
        auto logger = make_unique<MyLogger>(&inspector);

        CRSDistance distance(false);
        BandwidthWeight bandwidth(36, true, BandwidthWeight::Gaussian);
        SpatialWeight spatial(&bandwidth, &distance);

        GWSS algorithm;
        algorithm.setTelegram(std::move(logger));
        algorithm.setCoords(londonhp100_coord);
        algorithm.setVariables(x);
        algorithm.setSpatialWeight(spatial);
        algorithm.run();
        REQUIRE(inspector.progressed);
    }
}