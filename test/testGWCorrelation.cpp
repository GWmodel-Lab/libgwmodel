#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <vector>
#include <string>
#include <armadillo>
#include "gwmodelpp/GWCorrelation.h"
#include "gwmodelpp/spatialweight/CRSDistance.h"
#include "gwmodelpp/spatialweight/BandwidthWeight.h"
#include "gwmodelpp/spatialweight/SpatialWeight.h"
#include "londonhp100.h"
#include "TerminateCheckTelegram.h"

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif // ENABLE_OPENMP

using namespace std;
using namespace arma;
using namespace gwm;

TEST_CASE("GWCorrelation: londonhp100")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    SECTION("adaptive bandwidth | GWCorrelation | serial")
    {
        mat x = londonhp100_data.cols(2, 3);
        mat y = londonhp100_data.cols(0, 1);
        uword nVar = 4;
        vector<SpatialWeight> spatials;
        vector<GWCorrelation::BandwidthInitilizeType> bandwidthInitialize;
        // vector<GWCorrelation::BandwidthSelectionCriterionType> bandwidthSelectionApproach;
        for (size_t i = 0; i < nVar; i++)
        {
            CRSDistance distance;
            BandwidthWeight bandwidth(36, true, BandwidthWeight::Gaussian);
            spatials.push_back(SpatialWeight(&bandwidth, &distance));
            bandwidthInitialize.push_back(GWCorrelation::BandwidthInitilizeType::Specified);
        }

        GWCorrelation algorithm;
        algorithm.setCoords(londonhp100_coord);
        algorithm.setIndependentVariables(x);
        algorithm.setResponseVariables(y);
        algorithm.setSpatialWeights(spatials);
        algorithm.setBandwidthInitilize(bandwidthInitialize);
        // algorithm.setBandwidthSelectionApproach(bandwidthSelectionApproach);
        REQUIRE_NOTHROW(algorithm.run());
    }

    SECTION("adaptive bandwidth | GWCorrelation | serial")
    {

        mat x = londonhp100_data.cols(1, 3);
        mat y = londonhp100_data.col(0);
        uword nVar = 3;

        vector<SpatialWeight> spatials;
        vector<GWCorrelation::BandwidthInitilizeType> bandwidthInitialize;
        vector<GWCorrelation::BandwidthSelectionCriterionType> bandwidthSelectionApproach;
        for (size_t i = 0; i < nVar; i++)
        {
            CRSDistance distance;
            BandwidthWeight bandwidth(36, true, BandwidthWeight::Bisquare);
            spatials.push_back(SpatialWeight(&bandwidth, &distance));
            bandwidthInitialize.push_back(GWCorrelation::BandwidthInitilizeType::Null);
            bandwidthSelectionApproach.push_back(GWCorrelation::BandwidthSelectionCriterionType::CV);
        }

        GWCorrelation algorithm;
        algorithm.setCoords(londonhp100_coord);
        algorithm.setIndependentVariables(x);
        algorithm.setResponseVariables(y);
        algorithm.setSpatialWeights(spatials);
        algorithm.setBandwidthInitilize(bandwidthInitialize);
        algorithm.setBandwidthSelectionApproach(bandwidthSelectionApproach);
        REQUIRE_NOTHROW(algorithm.run());

    }

#ifdef ENABLE_OPENMP
    SECTION("adaptive bandwidth | GWCorrelation | omp parallel")
    {

        mat x = londonhp100_data.cols(1, 3);
        mat y = londonhp100_data.col(0);
        uword nVar = 3;

        vector<SpatialWeight> spatials;
        vector<GWCorrelation::BandwidthInitilizeType> bandwidthInitialize;
        vector<GWCorrelation::BandwidthSelectionCriterionType> bandwidthSelectionApproach;
        for (size_t i = 0; i < nVar; i++)
        {
            CRSDistance distance;
            BandwidthWeight bandwidth(36, true, BandwidthWeight::Bisquare);
            spatials.push_back(SpatialWeight(&bandwidth, &distance));
            bandwidthInitialize.push_back(GWCorrelation::BandwidthInitilizeType::Null);
            bandwidthSelectionApproach.push_back(GWCorrelation::BandwidthSelectionCriterionType::CV);
        }

        GWCorrelation algorithm;
        algorithm.setCoords(londonhp100_coord);
        algorithm.setIndependentVariables(x);
        algorithm.setResponseVariables(y);
        algorithm.setSpatialWeights(spatials);
        algorithm.setBandwidthInitilize(bandwidthInitialize);
        algorithm.setBandwidthSelectionApproach(bandwidthSelectionApproach);
        algorithm.setParallelType(ParallelType::OpenMP);
        algorithm.setOmpThreadNum(omp_get_num_threads());
        REQUIRE_NOTHROW(algorithm.run());

    }
#endif
}

