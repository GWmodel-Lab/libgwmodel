#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <vector>
#include <string>
#include <armadillo>
#include "gwmodelpp/CGwmSimpleLayer.h"
#include "gwmodelpp/CGwmRobustGWR.h"
#include "gwmodelpp/spatialweight/CGwmCRSDistance.h"
#include "gwmodelpp/spatialweight/CGwmBandwidthWeight.h"
#include "gwmodelpp/spatialweight/CGwmSpatialWeight.h"
#include "gwmodelpp/GwmVariable.h"
#include "londonhp100.h"

using namespace std;
using namespace arma;

vector<int> variables2indices(vector<GwmVariable> variables)
{
    vector<int> index(variables.size());
    std::transform(variables.begin(), variables.end(), index.begin(), [](const GwmVariable& v) -> int
    {
        return v.index;
    });
    return index;
}

TEST_CASE("RobustGWR: Filtered")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    CGwmSimpleLayer londonhp(londonhp100_coord, londonhp100_data, londonhp100_fields);
    REQUIRE(londonhp.points().n_rows);
    REQUIRE(londonhp.data().n_rows);
    REQUIRE(londonhp.fields().size());
    REQUIRE(londonhp.featureCount());

    CGwmCRSDistance distance(false);
    CGwmBandwidthWeight bandwidth(36, true, CGwmBandwidthWeight::Gaussian);
    CGwmSpatialWeight spatial(&bandwidth, &distance);

    GwmVariable purchase(0, true, "PURCHASE");
    GwmVariable floorsz(1, true, "FLOORSZ");
    GwmVariable unemploy(2, true, "UNEMPLOY");
    GwmVariable prof(3, true, "PROF");
    vector<GwmVariable> indepVars = { floorsz, unemploy, prof };

    CGwmRobustGWR algorithm;
    algorithm.setSourceLayer(&londonhp);
    algorithm.setDependentVariable(purchase);
    algorithm.setIndependentVariables(indepVars);
    algorithm.setSpatialWeight(spatial);
    algorithm.setHasHatMatrix(true);
    algorithm.setFiltered(true);
    REQUIRE_NOTHROW(algorithm.run());

    GwmRegressionDiagnostic diagnostic = algorithm.diagnostic();
    REQUIRE(abs(diagnostic.AIC - 2436.604) < 1e-3);
    REQUIRE(abs(diagnostic.AICc - 2448.272) < 1e-3);
    REQUIRE(abs(diagnostic.RSquare - 0.7080106) < 1e-3);
    REQUIRE(abs(diagnostic.RSquareAdjust - 0.6749753) < 1e-3);
}

TEST_CASE("RobustGWR: noFiltered")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    CGwmSimpleLayer londonhp(londonhp100_coord, londonhp100_data, londonhp100_fields);
    REQUIRE(londonhp.points().n_rows);
    REQUIRE(londonhp.data().n_rows);
    REQUIRE(londonhp.fields().size());
    REQUIRE(londonhp.featureCount());

    CGwmCRSDistance distance(false);
    CGwmBandwidthWeight bandwidth(36, true, CGwmBandwidthWeight::Gaussian);
    CGwmSpatialWeight spatial(&bandwidth, &distance);

    GwmVariable purchase(0, true, "PURCHASE");
    GwmVariable floorsz(1, true, "FLOORSZ");
    GwmVariable unemploy(2, true, "UNEMPLOY");
    GwmVariable prof(3, true, "PROF");
    vector<GwmVariable> indepVars = { floorsz, unemploy, prof };

    CGwmRobustGWR algorithm;
    algorithm.setSourceLayer(&londonhp);
    algorithm.setDependentVariable(purchase);
    algorithm.setIndependentVariables(indepVars);
    algorithm.setSpatialWeight(spatial);
    algorithm.setHasHatMatrix(true);
    algorithm.setFiltered(false);
    REQUIRE_NOTHROW(algorithm.run());

    GwmRegressionDiagnostic diagnostic = algorithm.diagnostic();
    REQUIRE(abs(diagnostic.AIC - 2436.604) < 1e-3);
    REQUIRE(abs(diagnostic.AICc - 2448.272) < 1e-3);
    REQUIRE(abs(diagnostic.RSquare - 0.7080106) < 1e-3);
    REQUIRE(abs(diagnostic.RSquareAdjust - 0.6749753) < 1e-3);
}

// TEST_CASE("RobustGWR: adaptive bandwidth autoselection of with CV")
// {
//     mat londonhp100_coord, londonhp100_data;
//     vector<string> londonhp100_fields;
//     if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
//     {
//         FAIL("Cannot load londonhp100 data.");
//     }

//     CGwmSimpleLayer londonhp(londonhp100_coord, londonhp100_data, londonhp100_fields);
//     REQUIRE(londonhp.points().n_rows);
//     REQUIRE(londonhp.data().n_rows);
//     REQUIRE(londonhp.fields().size());
//     REQUIRE(londonhp.featureCount());

//     CGwmCRSDistance distance(false);
//     CGwmBandwidthWeight bandwidth(0, true, CGwmBandwidthWeight::Gaussian);
//     CGwmSpatialWeight spatial(&bandwidth, &distance);

//     GwmVariable purchase(0, true, "PURCHASE");
//     GwmVariable floorsz(1, true, "FLOORSZ");
//     GwmVariable unemploy(2, true, "UNEMPLOY");
//     GwmVariable prof(3, true, "PROF");
//     vector<GwmVariable> indepVars = { floorsz, unemploy, prof };

//     CGwmRobustGWR algorithm;
//     algorithm.setSourceLayer(&londonhp);
//     algorithm.setDependentVariable(purchase);
//     algorithm.setIndependentVariables(indepVars);
//     algorithm.setSpatialWeight(spatial);
//     algorithm.setHasHatMatrix(true);

//     algorithm.setIsAutoselectBandwidth(true);
//     algorithm.setBandwidthSelectionCriterion(CGwmRobustGWR::BandwidthSelectionCriterionType::CV);
    
//     REQUIRE_NOTHROW(algorithm.run());

//     int bw = algorithm.spatialWeight().weight<CGwmBandwidthWeight>()->bandwidth();
//     REQUIRE(bw == 67);
// }

// TEST_CASE("RobustGWR: indepdent variable autoselection with AIC")
// {
//     mat londonhp100_coord, londonhp100_data;
//     vector<string> londonhp100_fields;
//     if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
//     {
//         FAIL("Cannot load londonhp100 data.");
//     }

//     CGwmSimpleLayer londonhp(londonhp100_coord, londonhp100_data, londonhp100_fields);
//     REQUIRE(londonhp.points().n_rows);
//     REQUIRE(londonhp.data().n_rows);
//     REQUIRE(londonhp.fields().size());
//     REQUIRE(londonhp.featureCount());

//     CGwmCRSDistance distance(false);
//     CGwmBandwidthWeight bandwidth(36, true, CGwmBandwidthWeight::Gaussian);
//     CGwmSpatialWeight spatial(&bandwidth, &distance);

//     GwmVariable purchase(0, true, "PURCHASE");
//     GwmVariable floorsz(1, true, "FLOORSZ");
//     GwmVariable unemploy(2, true, "UNEMPLOY");
//     GwmVariable prof(3, true, "PROF");
//     vector<GwmVariable> indepVars = { floorsz, unemploy, prof };

//     CGwmRobustGWR algorithm;
//     algorithm.setSourceLayer(&londonhp);
//     algorithm.setDependentVariable(purchase);
//     algorithm.setIndependentVariables(indepVars);
//     algorithm.setSpatialWeight(spatial);
//     algorithm.setHasHatMatrix(true);

//     algorithm.setIsAutoselectIndepVars(true);
//     algorithm.setIndepVarSelectionThreshold(3.0);
    
//     REQUIRE_NOTHROW(algorithm.run());

//     VariablesCriterionList criterions = algorithm.indepVarsSelectionCriterionList();
//     REQUIRE_THAT(variables2indices(criterions[0].first), Catch::Equals(vector<int>({ 2 })));
//     REQUIRE_THAT(criterions[0].second, Catch::WithinAbs(2551.61359020599, 1e-8));
//     REQUIRE_THAT(variables2indices(criterions[1].first), Catch::Equals(vector<int>({ 3 })));
//     REQUIRE_THAT(criterions[1].second, Catch::WithinAbs(2551.30032201349, 1e-8));
//     REQUIRE_THAT(variables2indices(criterions[2].first), Catch::Equals(vector<int>({ 1 })));
//     REQUIRE_THAT(criterions[2].second, Catch::WithinAbs(2468.93236280013, 1e-8));
//     REQUIRE_THAT(variables2indices(criterions[3].first), Catch::Equals(vector<int>({ 1, 3 })));
//     REQUIRE_THAT(criterions[3].second, Catch::WithinAbs(2452.86447942033, 1e-8));
//     REQUIRE_THAT(variables2indices(criterions[4].first), Catch::Equals(vector<int>({ 1, 2 })));
//     REQUIRE_THAT(criterions[4].second, Catch::WithinAbs(2450.59642666509, 1e-8));
//     REQUIRE_THAT(variables2indices(criterions[5].first), Catch::Equals(vector<int>({ 1, 2, 3 })));
//     REQUIRE_THAT(criterions[5].second, Catch::WithinAbs(2452.80388934625, 1e-8));
// }

// TEST_CASE("RobustGWR: multithread basic flow")
// {
//     mat londonhp100_coord, londonhp100_data;
//     vector<string> londonhp100_fields;
//     if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
//     {
//         FAIL("Cannot load londonhp100 data.");
//     }

//     CGwmSimpleLayer londonhp(londonhp100_coord, londonhp100_data, londonhp100_fields);
//     REQUIRE(londonhp.points().n_rows);
//     REQUIRE(londonhp.data().n_rows);
//     REQUIRE(londonhp.fields().size());
//     REQUIRE(londonhp.featureCount());

//     CGwmCRSDistance distance(false);
//     CGwmBandwidthWeight bandwidth(36, true, CGwmBandwidthWeight::Gaussian);
//     CGwmSpatialWeight spatial(&bandwidth, &distance);

//     GwmVariable purchase(0, true, "PURCHASE");
//     GwmVariable floorsz(1, true, "FLOORSZ");
//     GwmVariable unemploy(2, true, "UNEMPLOY");
//     GwmVariable prof(3, true, "PROF");
//     vector<GwmVariable> indepVars = { floorsz, unemploy, prof };

//     CGwmRobustGWR algorithm;
//     algorithm.setSourceLayer(&londonhp);
//     algorithm.setDependentVariable(purchase);
//     algorithm.setIndependentVariables(indepVars);
//     algorithm.setSpatialWeight(spatial);
//     algorithm.setHasHatMatrix(true);
//     algorithm.setIsAutoselectBandwidth(true);
//     algorithm.setBandwidthSelectionCriterion(CGwmGWRBasic::BandwidthSelectionCriterionType::CV);
//     algorithm.setIsAutoselectIndepVars(true);
//     algorithm.setIndepVarSelectionThreshold(3.0);
//     algorithm.setParallelType(ParallelType::OpenMP);
//     algorithm.setOmpThreadNum(6);
//     REQUIRE_NOTHROW(algorithm.run());

//     VariablesCriterionList criterions = algorithm.indepVarsSelectionCriterionList();
//     REQUIRE_THAT(variables2indices(criterions[0].first), Catch::Equals(vector<int>({ 2 })));
//     REQUIRE_THAT(criterions[0].second, Catch::WithinAbs(2551.61359020599, 1e-8));
//     REQUIRE_THAT(variables2indices(criterions[1].first), Catch::Equals(vector<int>({ 3 })));
//     REQUIRE_THAT(criterions[1].second, Catch::WithinAbs(2551.30032201349, 1e-8));
//     REQUIRE_THAT(variables2indices(criterions[2].first), Catch::Equals(vector<int>({ 1 })));
//     REQUIRE_THAT(criterions[2].second, Catch::WithinAbs(2468.93236280013, 1e-8));
//     REQUIRE_THAT(variables2indices(criterions[3].first), Catch::Equals(vector<int>({ 1, 3 })));
//     REQUIRE_THAT(criterions[3].second, Catch::WithinAbs(2452.86447942033, 1e-8));
//     REQUIRE_THAT(variables2indices(criterions[4].first), Catch::Equals(vector<int>({ 1, 2 })));
//     REQUIRE_THAT(criterions[4].second, Catch::WithinAbs(2450.59642666509, 1e-8));
//     REQUIRE_THAT(variables2indices(criterions[5].first), Catch::Equals(vector<int>({ 1, 2, 3 })));
//     REQUIRE_THAT(criterions[5].second, Catch::WithinAbs(2452.80388934625, 1e-8));

//     double bw = algorithm.spatialWeight().weight<CGwmBandwidthWeight>()->bandwidth();
//     REQUIRE(bw == 31.0);

//     GwmRegressionDiagnostic diagnostic = algorithm.diagnostic();
//     REQUIRE(abs(diagnostic.AIC - 2435.8161441795) < 1e-8);
//     REQUIRE(abs(diagnostic.AICc - 2445.49629974057) < 1e-8);
//     REQUIRE(abs(diagnostic.RSquare - 0.706143867720706) < 1e-8);
//     REQUIRE(abs(diagnostic.RSquareAdjust - 0.678982114793865) < 1e-8);
// }
