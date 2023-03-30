#define CATCH_CONFIG_MAIN

#include <catch2/catch_all.hpp>

#include <vector>
#include <string>
#include <armadillo>


#include "gwmodelpp/spatialweight/CRSSTDistance.h"
#include "gwmodelpp/spatialweight/BandwidthWeight.h"
#include "gwmodelpp/spatialweight/SpatialWeight.h"
#include "gwmodelpp/GTWR.h"

#include "include/londonhp100.h"

using namespace std;
using namespace arma;
using namespace gwm;

TEST_CASE("GTWR: londonhp100")
{
    mat londonhp100_coord, londonhp100_data;
    vec londonhp100_times;
    vector<string> londonhp100_fields;
    if (!read_londonhp100temporal(londonhp100_coord, londonhp100_times, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100temporal data.");
    }

    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_coord.n_rows), londonhp100_data.cols(1, 3));

    GTWR algorithm;
    algorithm.setCoords(londonhp100_coord,londonhp100_times);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    CRSDistance sdist(false);
    OneDimDistance tdist;

    SECTION("adaptive bandwidth | no bandwidth optimization | lambda=1 ") {
        CRSSTDistance distance(&sdist, &tdist, 1);
        BandwidthWeight bandwidth(36,true, BandwidthWeight::Gaussian);
        SpatialWeight spatial(&bandwidth, &distance);
        algorithm.setSpatialWeight(spatial);
        algorithm.setHasHatMatrix(true);
        REQUIRE_NOTHROW(algorithm.fit());
        RegressionDiagnostic diagnostic = algorithm.diagnostic();
        REQUIRE(algorithm.hasIntercept() == true);
        ////lambda=0 
        // REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(2443.9698348782, 1e-8));
        // REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2456.3750569354, 1e-8));
        // REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.6872921780938, 1e-8));
        // REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.65184964517969, 1e-8));
        ////lambda=1
        REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(2436.6044573089, 1e-8));
        REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2448.2720652516, 1e-8));
        REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.7080106320292, 1e-8));
        REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.67497534170905, 1e-8));
        ////lambda=0.05
        // REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(2440.2982293126, 1e-8));
        // REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2453.0875630431 , 1e-8));
        // REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.69935365208496 , 1e-8));
        // REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.66296885453873, 1e-8));
    }
    SECTION("fixed bandwidth | no bandwidth optimization | lambda=1 ") {
        CRSSTDistance distance(&sdist, &tdist, 1);
        BandwidthWeight bandwidth(5000,false, BandwidthWeight::Gaussian);
        SpatialWeight spatial(&bandwidth, &distance);
        algorithm.setSpatialWeight(spatial);
        algorithm.setHasHatMatrix(true);
        REQUIRE_NOTHROW(algorithm.fit());
        RegressionDiagnostic diagnostic = algorithm.diagnostic();
        REQUIRE(algorithm.hasIntercept() == true);
        //lambda=0 
        // REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(2443.9698348782, 1e-8));
        // REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2456.3750569354, 1e-8));
        // REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.6872921780938, 1e-8));
        // REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.65184964517969, 1e-8));
        //lambda=1
        REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(2437.6495742714, 1e-8));
        REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2447.6762811677, 1e-8));
        REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.70146629544468, 1e-8));
        REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.67369159445212, 1e-8));
        //lambda=0.05
        // REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(2440.2982293126, 1e-8));
        // REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2453.0875630431 , 1e-8));
        // REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.69935365208496 , 1e-8));
        // REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.66296885453873, 1e-8));
    }
    SECTION("adaptive bandwidth | CV bandwidth optimization | lambda=0 ") {
        CRSSTDistance distance(&sdist, &tdist, 0.0);
        BandwidthWeight bandwidth(0, true, BandwidthWeight::Gaussian);
        SpatialWeight spatial(&bandwidth, &distance);
        algorithm.setSpatialWeight(spatial);
        algorithm.setIsAutoselectBandwidth(true);
        algorithm.setBandwidthSelectionCriterion(GTWR::BandwidthSelectionCriterionType::CV);
        REQUIRE_NOTHROW(algorithm.fit());
        size_t bw = (size_t)algorithm.spatialWeight().weight<BandwidthWeight>()->bandwidth();
        REQUIRE(bw == 94);
    }
    SECTION("fixed bandwidth | CV bandwidth optimization | lambda=1 ") {
        CRSSTDistance distance(&sdist, &tdist, 1);
        BandwidthWeight bandwidth(0, false, BandwidthWeight::Gaussian);
        SpatialWeight spatial(&bandwidth, &distance);
        algorithm.setSpatialWeight(spatial);
        algorithm.setIsAutoselectBandwidth(true);
        algorithm.setBandwidthSelectionCriterion(GTWR::BandwidthSelectionCriterionType::CV);
        REQUIRE_NOTHROW(algorithm.fit());
        size_t bw = (size_t)algorithm.spatialWeight().weight<BandwidthWeight>()->bandwidth();
        REQUIRE(bw == 5076);
    }
}