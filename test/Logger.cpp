#define CATCH_CONFIG_MAIN

#include <catch2/catch_all.hpp>

#include <vector>
#include <string>
#include <armadillo>
#include "gwmodelpp/CGwmGWRMultiscale.h"

#include "gwmodelpp/spatialweight/CGwmCRSDistance.h"
#include "gwmodelpp/spatialweight/CGwmBandwidthWeight.h"
#include "gwmodelpp/spatialweight/CGwmSpatialWeight.h"
#include "gwmodelpp/GwmLogger.h"
#include "londonhp100.h"

using namespace std;
using namespace arma;
using namespace gwm;

void printer(string message, GwmLogger::LogLevel level, string fun_name, string file_name)
{
    cout << "[" << fun_name << "] (in file " << file_name << "): " << message << "\n";
}

TEST_CASE("MGWR: basic flow")
{
    GwmLogger::logger = printer;

    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    uword nVar = 3;
    vector<CGwmSpatialWeight> spatials;
    vector<bool> preditorCentered;
    vector<CGwmGWRMultiscale::BandwidthInitilizeType> bandwidthInitialize;
    vector<CGwmGWRMultiscale::BandwidthSelectionCriterionType> bandwidthSelectionApproach;
    for (size_t i = 0; i < nVar; i++)
    {
        CGwmCRSDistance distance;
        CGwmBandwidthWeight bandwidth(0, false, CGwmBandwidthWeight::Bisquare);
        spatials.push_back(CGwmSpatialWeight(&bandwidth, &distance));
        preditorCentered.push_back(i != 0);
        bandwidthInitialize.push_back(CGwmGWRMultiscale::BandwidthInitilizeType::Null);
        bandwidthSelectionApproach.push_back(CGwmGWRMultiscale::BandwidthSelectionCriterionType::CV);
    }

    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_data.n_rows), londonhp100_data.cols(1, 3));

    CGwmGWRMultiscale algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeights(spatials);
    algorithm.setHasHatMatrix(true);
    REQUIRE_THROWS(algorithm.setBandwidthInitilize(bandwidthInitialize));
}