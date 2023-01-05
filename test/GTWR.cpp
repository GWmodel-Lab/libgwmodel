#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>

#include <vector>
#include <string>
#include <armadillo>


#include "gwmodelpp/spatialweight/CGwmCRSSTDistance.h"
#include "gwmodelpp/spatialweight/CGwmBandwidthWeight.h"
#include "gwmodelpp/spatialweight/CGwmSpatialWeight.h"
#include "gwmodelpp/CGwmGTWR.h"

#include "include/londonhp100.h"

using namespace std;
using namespace arma;

TEST_CASE("GTWR: basic flow")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100temporal(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100temporal data.");
    }

    CGwmCRSSTDistance distance(false,0.05);
    CGwmBandwidthWeight bandwidth(50, true, CGwmBandwidthWeight::Gaussian);
    CGwmSpatialWeight spatial(&bandwidth, &distance);

    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_coord.n_rows), londonhp100_data.cols(1, 3));

    CGwmGTWR algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeight(spatial);
    algorithm.setHasHatMatrix(true);
    REQUIRE_NOTHROW(algorithm.fit());

    GwmRegressionDiagnostic diagnostic = algorithm.diagnostic();
    REQUIRE_THAT(diagnostic.AIC, Catch::WithinAbs(2441.907430935, 1e-8));
    REQUIRE_THAT(diagnostic.AICc, Catch::WithinAbs(2453.5153601032, 1e-8));
    REQUIRE_THAT(diagnostic.RSquare, Catch::WithinAbs(0.67006467896073, 1e-8));
    REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::WithinAbs(0.65086284199265 , 1e-8));

    REQUIRE(algorithm.hasIntercept() == true);
}
