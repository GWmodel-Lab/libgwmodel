#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <vector>
#include <string>
#include <armadillo>
#include "gwmodelpp/CGwmGWRBasic.h"
#include "gwmodelpp/spatialweight/CGwmCRSDistance.h"
#include "gwmodelpp/spatialweight/CGwmBandwidthWeight.h"
#include "gwmodelpp/spatialweight/CGwmSpatialWeight.h"
#include "londonhp100.h"

using namespace std;
using namespace arma;

TEST_CASE("BasicGWR: basic flow")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    CGwmCRSDistance distance(false);
    CGwmBandwidthWeight bandwidth(36, true, CGwmBandwidthWeight::Gaussian);
    CGwmSpatialWeight spatial(&bandwidth, &distance);

    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_coord.n_rows), londonhp100_data.cols(1, 3));

    CGwmGWRBasic algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeight(spatial);
    algorithm.setHasHatMatrix(true);
    REQUIRE_NOTHROW(algorithm.fit());

    GwmRegressionDiagnostic diagnostic = algorithm.diagnostic();
    REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(2436.60445730413, 1e-8));
    REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2448.27206524754, 1e-8));
    REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.708010632044736, 1e-8));
    REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.674975341723766, 1e-8));

    REQUIRE(algorithm.hasIntercept() == true);
}

TEST_CASE("BasicGWR: adaptive bandwidth autoselection of with CV")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    CGwmCRSDistance distance(false);
    CGwmBandwidthWeight bandwidth(0, true, CGwmBandwidthWeight::Gaussian);
    CGwmSpatialWeight spatial(&bandwidth, &distance);

    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_coord.n_rows), londonhp100_data.cols(1, 3));

    CGwmGWRBasic algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeight(spatial);
    algorithm.setHasHatMatrix(true);

    algorithm.setIsAutoselectBandwidth(true);
    algorithm.setBandwidthSelectionCriterion(CGwmGWRBasic::BandwidthSelectionCriterionType::CV);
    
    REQUIRE_NOTHROW(algorithm.fit());

    size_t bw = (size_t)algorithm.spatialWeight().weight<CGwmBandwidthWeight>()->bandwidth();
    REQUIRE(bw == 67);
}

TEST_CASE("BasicGWR: indepdent variable autoselection with AIC")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    CGwmCRSDistance distance(false);
    CGwmBandwidthWeight bandwidth(36, true, CGwmBandwidthWeight::Gaussian);
    CGwmSpatialWeight spatial(&bandwidth, &distance);

    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_coord.n_rows), londonhp100_data.cols(1, 3));

    CGwmGWRBasic algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeight(spatial);
    algorithm.setHasHatMatrix(true);

    algorithm.setIsAutoselectIndepVars(true);
    algorithm.setIndepVarSelectionThreshold(3.0);
    
    REQUIRE_NOTHROW(algorithm.fit());

    VariablesCriterionList criterions = algorithm.indepVarsSelectionCriterionList();
    REQUIRE_THAT(criterions[0].first, Catch::Matchers::Equals(vector<size_t>({ 2 })));
    REQUIRE_THAT(criterions[0].second, Catch::Matchers::WithinAbs(2551.61359020599, 1e-8));
    REQUIRE_THAT(criterions[1].first, Catch::Matchers::Equals(vector<size_t>({ 3 })));
    REQUIRE_THAT(criterions[1].second, Catch::Matchers::WithinAbs(2551.30032201349, 1e-8));
    REQUIRE_THAT(criterions[2].first, Catch::Matchers::Equals(vector<size_t>({ 1 })));
    REQUIRE_THAT(criterions[2].second, Catch::Matchers::WithinAbs(2468.93236280013, 1e-8));
    REQUIRE_THAT(criterions[3].first, Catch::Matchers::Equals(vector<size_t>({ 1, 3 })));
    REQUIRE_THAT(criterions[3].second, Catch::Matchers::WithinAbs(2452.86447942033, 1e-8));
    REQUIRE_THAT(criterions[4].first, Catch::Matchers::Equals(vector<size_t>({ 1, 2 })));
    REQUIRE_THAT(criterions[4].second, Catch::Matchers::WithinAbs(2450.59642666509, 1e-8));
    REQUIRE_THAT(criterions[5].first, Catch::Matchers::Equals(vector<size_t>({ 1, 2, 3 })));
    REQUIRE_THAT(criterions[5].second, Catch::Matchers::WithinAbs(2452.80388934625, 1e-8));

    vector<size_t> selectedVariables = algorithm.selectedVariables();
    REQUIRE_THAT(selectedVariables, Catch::Matchers::Equals(vector<size_t>({1, 3})));
}

#ifdef ENABLE_OPENMP
TEST_CASE("BasicGWR: multithread basic flow")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    CGwmCRSDistance distance(false);
    CGwmBandwidthWeight bandwidth(36, true, CGwmBandwidthWeight::Gaussian);
    CGwmSpatialWeight spatial(&bandwidth, &distance);

    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_coord.n_rows), londonhp100_data.cols(1, 3));

    CGwmGWRBasic algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeight(spatial);
    algorithm.setHasHatMatrix(true);
    algorithm.setIsAutoselectBandwidth(true);
    algorithm.setBandwidthSelectionCriterion(CGwmGWRBasic::BandwidthSelectionCriterionType::CV);
    algorithm.setIsAutoselectIndepVars(true);
    algorithm.setIndepVarSelectionThreshold(3.0);
    algorithm.setParallelType(ParallelType::OpenMP);
    algorithm.setOmpThreadNum(6);
    REQUIRE_NOTHROW(algorithm.fit());

    VariablesCriterionList criterions = algorithm.indepVarsSelectionCriterionList();
    REQUIRE_THAT(criterions[0].first, Catch::Matchers::Equals(vector<size_t>({ 2 })));
    REQUIRE_THAT(criterions[0].second, Catch::Matchers::WithinAbs(2551.61359020599, 1e-8));
    REQUIRE_THAT(criterions[1].first, Catch::Matchers::Equals(vector<size_t>({ 3 })));
    REQUIRE_THAT(criterions[1].second, Catch::Matchers::WithinAbs(2551.30032201349, 1e-8));
    REQUIRE_THAT(criterions[2].first, Catch::Matchers::Equals(vector<size_t>({ 1 })));
    REQUIRE_THAT(criterions[2].second, Catch::Matchers::WithinAbs(2468.93236280013, 1e-8));
    REQUIRE_THAT(criterions[3].first, Catch::Matchers::Equals(vector<size_t>({ 1, 3 })));
    REQUIRE_THAT(criterions[3].second, Catch::Matchers::WithinAbs(2452.86447942033, 1e-8));
    REQUIRE_THAT(criterions[4].first, Catch::Matchers::Equals(vector<size_t>({ 1, 2 })));
    REQUIRE_THAT(criterions[4].second, Catch::Matchers::WithinAbs(2450.59642666509, 1e-8));
    REQUIRE_THAT(criterions[5].first, Catch::Matchers::Equals(vector<size_t>({ 1, 2, 3 })));
    REQUIRE_THAT(criterions[5].second, Catch::Matchers::WithinAbs(2452.80388934625, 1e-8));
    
    vector<size_t> selectedVariables = algorithm.selectedVariables();
    REQUIRE_THAT(selectedVariables, Catch::Matchers::Equals(vector<size_t>({1, 3})));

    double bw = algorithm.spatialWeight().weight<CGwmBandwidthWeight>()->bandwidth();
    REQUIRE(bw == 31.0);

    GwmRegressionDiagnostic diagnostic = algorithm.diagnostic();
    REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(2435.8161441795, 1e-8));
    REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2445.49629974057, 1e-8));
    REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.706143867720706, 1e-8));
    REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.678982114793865, 1e-8));
}
#endif
