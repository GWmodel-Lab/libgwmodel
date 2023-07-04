#define CATCH_CONFIG_MAIN

#include <catch2/catch_all.hpp>

#include <vector>
#include <string>
#include <armadillo>
#include "gwmodelpp/GWRMultiscale.h"

#include "gwmodelpp/spatialweight/CRSDistance.h"
#include "gwmodelpp/spatialweight/BandwidthWeight.h"
#include "gwmodelpp/spatialweight/SpatialWeight.h"
#include "gwmodelpp/Logger.h"
#include "londonhp100.h"

using namespace std;
using namespace arma;
using namespace gwm;

struct MyLogger : Logger
{
    ~MyLogger()
    {
        cout << "[MyLogger] Logger destructed!";
    }

    void print(string message, ITelegram::LogLevel level, string fun_name, string file_name) override
    {
        cout << "[" << fun_name << "] (in file " << file_name << "): " << message << "\n";
        called = true;
    }

    bool called = false;
};

TEST_CASE("MGWR: basic flow")
{
    MyLogger* logger = new MyLogger();
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

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

    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_data.n_rows), londonhp100_data.cols(1, 3));

    GWRMultiscale algorithm;
    algorithm.setTelegram(logger);
    algorithm.setCoords(londonhp100_coord);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeights(spatials);
    algorithm.setHasHatMatrix(true);
    algorithm.setBandwidthInitilize(bandwidthInitialize);

    REQUIRE(logger->called);
}