#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <vector>
#include <string>
#include <armadillo>
#include "gwmodelpp/CGwmGWRScalable.h"
#include "gwmodelpp/spatialweight/CGwmCRSDistance.h"
#include "gwmodelpp/spatialweight/CGwmBandwidthWeight.h"
#include "gwmodelpp/spatialweight/CGwmSpatialWeight.h"
#include "londonhp100.h"
using namespace std;
using namespace arma;

TEST_CASE("ScalableGWR: basic flow")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    CGwmCRSDistance distance(false);
    CGwmBandwidthWeight bandwidth(60, true, CGwmBandwidthWeight::Gaussian);
    CGwmSpatialWeight spatial(&bandwidth, &distance);

    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_coord.n_rows), londonhp100_data.cols(1, 3));

    CGwmGWRScalable algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeight(spatial);
    algorithm.setPolynomial(4);
    algorithm.setHasHatMatrix(true);
    algorithm.setParameterOptimizeCriterion(CGwmGWRScalable::BandwidthSelectionCriterionType::CV);
    //algorithm.setPolynomial();
    REQUIRE_NOTHROW(algorithm.fit());

    GwmRegressionDiagnostic diagnostic = algorithm.diagnostic();
    REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(2454.93832121, 1e-3));
    REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2548.83573789, 1e-3));
    REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.7418439605, 1e-3));
    REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.7730231604, 1e-3));

    REQUIRE(algorithm.hasIntercept() == true);
}

TEST_CASE("ScalableGWR:  bandwidth  of with AIC")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    CGwmCRSDistance distance(false);
    CGwmBandwidthWeight bandwidth(60, true, CGwmBandwidthWeight::Gaussian);
    CGwmSpatialWeight spatial(&bandwidth, &distance);

    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_coord.n_rows), londonhp100_data.cols(1, 3));

    CGwmGWRScalable algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeight(spatial);
    algorithm.setHasHatMatrix(true);

    //algorithm.setIsAutoselectBandwidth(true);
    algorithm.setParameterOptimizeCriterion(CGwmGWRScalable::BandwidthSelectionCriterionType::AIC);
    
    REQUIRE_NOTHROW(algorithm.fit());

    GwmRegressionDiagnostic diagnostic = algorithm.diagnostic();
    /*REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(2437.3887978, 1e-3));
    REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2445.49861419, 1e-3));
    REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.6979320133, 1e-3));
    REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.7211781422, 1e-3));*/

    REQUIRE(algorithm.hasIntercept() == true);
    
}

TEST_CASE("ScableGWR:  autoselection with CV")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    CGwmCRSDistance distance(false);
    CGwmBandwidthWeight bandwidth(60, true, CGwmBandwidthWeight::Gaussian);
    CGwmSpatialWeight spatial(&bandwidth, &distance);

    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_coord.n_rows), londonhp100_data.cols(1, 3));

    CGwmGWRScalable algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeight(spatial);
    algorithm.setHasHatMatrix(true);
    
    REQUIRE_NOTHROW(algorithm.fit());

    GwmRegressionDiagnostic diagnostic = algorithm.diagnostic();
    REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(2454.93832121, 1e-3));
    REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2548.83573789, 1e-3));
    REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.7418439605, 1e-3));
    REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.7730231604, 1e-3));

    REQUIRE(algorithm.hasIntercept() == true);
    
}
