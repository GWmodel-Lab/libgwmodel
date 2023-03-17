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

TEST_CASE("GTWR: basic flow")
{
    mat londonhp100_coord, londonhp100_data;
    vec londonhp100_times;
    vector<string> londonhp100_fields;
    if (!read_londonhp100temporal(londonhp100_coord, londonhp100_times, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100temporal data.");
    }

    CRSDistance sdist(false);
    OneDimDistance tdist;
    CRSSTDistance distance(&sdist, &tdist, 0);
    BandwidthWeight bandwidth(36,true, BandwidthWeight::Gaussian);
    SpatialWeight spatial(&bandwidth, &distance);

    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_coord.n_rows), londonhp100_data.cols(1, 3));

    GTWR algorithm;
    algorithm.setCoords(londonhp100_coord,londonhp100_times);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeight(spatial);
    algorithm.setHasHatMatrix(true);
    REQUIRE_NOTHROW(algorithm.fit());

    RegressionDiagnostic diagnostic = algorithm.diagnostic();
    // //these are diagnostics when lambda=0 
    REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(2443.9698348782, 1e-8));
    REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2456.3750569354, 1e-8));
    REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.6872921780938, 1e-8));
    REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.65184964517969, 1e-8));
    //lambda=1
    // REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(2436.60445730413, 1e-8));
    // REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2448.27206524754, 1e-8));
    // REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.708010632044736, 1e-8));
    // REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.674975341723766, 1e-8));

    //lambda=0.05
    // REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(2440.2982293126, 1e-8));
    // REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2453.0875630431 , 1e-8));
    // REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.69935365208496 , 1e-8));
    // REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.66296885453873, 1e-8));

    REQUIRE(algorithm.hasIntercept() == true);
}