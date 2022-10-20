#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>

#include <vector>
#include <string>
#include <armadillo>
#include "gwmodelpp/CGwmSimpleLayer.h"
#include "gwmodelpp/CGwmGWDR.h"

#include "gwmodelpp/spatialweight/CGwmOneDimDistance.h"
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

TEST_CASE("GWDR: basic flow")
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

    uword nDim = londonhp.points().n_cols;
    vector<CGwmSpatialWeight> spatials;
    for (size_t i = 0; i < nDim; i++)
    {
        CGwmOneDimDistance distance;
        CGwmBandwidthWeight bandwidth(36, true, CGwmBandwidthWeight::Bisquare);
        spatials.push_back(CGwmSpatialWeight(&bandwidth, &distance));
    }

    GwmVariable purchase(0, true, "PURCHASE");
    GwmVariable floorsz(1, true, "FLOORSZ");
    GwmVariable unemploy(2, true, "UNEMPLOY");
    GwmVariable prof(3, true, "PROF");
    vector<GwmVariable> indepVars = { floorsz, unemploy, prof };

    CGwmGWDR algorithm;
    algorithm.setSourceLayer(&londonhp);
    algorithm.setDependentVariable(purchase);
    algorithm.setIndependentVariables(indepVars);
    algorithm.setSpatialWeights(spatials);
    algorithm.setHasHatMatrix(true);
    REQUIRE_NOTHROW(algorithm.run());

    GwmRegressionDiagnostic diagnostic = algorithm.diagnostic();
    REQUIRE_THAT(diagnostic.AICc, Catch::WithinAbs(2580.754861403243, 1e-6));
    REQUIRE_THAT(diagnostic.RSquare, Catch::WithinAbs(0.898063766825, 1e-6));
    REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::WithinAbs(0.722050880566, 1e-6));
}

TEST_CASE("GWDR: basic flow with bandwidth optimization (CV)")
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

    uword nDim = londonhp.points().n_cols;
    vector<CGwmSpatialWeight> spatials;
    for (size_t i = 0; i < nDim; i++)
    {
        CGwmOneDimDistance distance;
        CGwmBandwidthWeight bandwidth(0, true, CGwmBandwidthWeight::Bisquare);
        spatials.push_back(CGwmSpatialWeight(&bandwidth, &distance));
    }

    GwmVariable purchase(0, true, "PURCHASE");
    GwmVariable floorsz(1, true, "FLOORSZ");
    GwmVariable unemploy(2, true, "UNEMPLOY");
    GwmVariable prof(3, true, "PROF");
    vector<GwmVariable> indepVars = { floorsz, unemploy, prof };

    CGwmGWDR algorithm;
    algorithm.setSourceLayer(&londonhp);
    algorithm.setDependentVariable(purchase);
    algorithm.setIndependentVariables(indepVars);
    algorithm.setSpatialWeights(spatials);
    algorithm.setEnableBandwidthOptimize(true);
    algorithm.setBandwidthOptimizeStep(0.01);
    algorithm.setBandwidthCriterionType(CGwmGWDR::CV);
    algorithm.setHasHatMatrix(true);
    REQUIRE_NOTHROW(algorithm.run());

    const vector<CGwmSpatialWeight>& spatialWeights = algorithm.spatialWeights();
    REQUIRE_THAT(spatialWeights[0].weight<CGwmBandwidthWeight>()->bandwidth(), Catch::WithinAbs(80, 1e-12));
    REQUIRE_THAT(spatialWeights[1].weight<CGwmBandwidthWeight>()->bandwidth(), Catch::WithinAbs(2550816, 1e-12));
}

TEST_CASE("GWDR: basic flow with bandwidth optimization (AIC)")
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

    uword nDim = londonhp.points().n_cols;
    vector<CGwmSpatialWeight> spatials;
    for (size_t i = 0; i < nDim; i++)
    {
        CGwmOneDimDistance distance;
        CGwmBandwidthWeight bandwidth(0, true, CGwmBandwidthWeight::Bisquare);
        spatials.push_back(CGwmSpatialWeight(&bandwidth, &distance));
    }

    GwmVariable purchase(0, true, "PURCHASE");
    GwmVariable floorsz(1, true, "FLOORSZ");
    GwmVariable unemploy(2, true, "UNEMPLOY");
    GwmVariable prof(3, true, "PROF");
    vector<GwmVariable> indepVars = { floorsz, unemploy, prof };

    CGwmGWDR algorithm;
    algorithm.setSourceLayer(&londonhp);
    algorithm.setDependentVariable(purchase);
    algorithm.setIndependentVariables(indepVars);
    algorithm.setSpatialWeights(spatials);
    algorithm.setEnableBandwidthOptimize(true);
    algorithm.setBandwidthCriterionType(CGwmGWDR::AIC);
    algorithm.setHasHatMatrix(true);
    REQUIRE_NOTHROW(algorithm.run());

    const vector<CGwmSpatialWeight>& spatialWeights = algorithm.spatialWeights();
    REQUIRE_THAT(spatialWeights[0].weight<CGwmBandwidthWeight>()->bandwidth(), Catch::WithinAbs(80, 1e-12));
    REQUIRE_THAT(spatialWeights[1].weight<CGwmBandwidthWeight>()->bandwidth(), Catch::WithinAbs(189, 1e-12));
}

TEST_CASE("GWDR: basic flow with independent variable selection")
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

    uword nDim = londonhp.points().n_cols;
    vector<CGwmSpatialWeight> spatials;
    for (size_t i = 0; i < nDim; i++)
    {
        CGwmOneDimDistance distance;
        CGwmBandwidthWeight bandwidth(0.618 * londonhp.featureCount(), true, CGwmBandwidthWeight::Bisquare);
        spatials.push_back(CGwmSpatialWeight(&bandwidth, &distance));
    }

    GwmVariable purchase(0, true, "PURCHASE");
    GwmVariable floorsz(1, true, "FLOORSZ");
    GwmVariable unemploy(2, true, "UNEMPLOY");
    GwmVariable prof(3, true, "PROF");
    vector<GwmVariable> indepVars = { floorsz, unemploy, prof };

    CGwmGWDR algorithm;
    algorithm.setSourceLayer(&londonhp);
    algorithm.setDependentVariable(purchase);
    algorithm.setIndependentVariables(indepVars);
    algorithm.setSpatialWeights(spatials);
    algorithm.setEnableIndepVarSelect(true);
    algorithm.setBandwidthCriterionType(CGwmGWDR::AIC);
    algorithm.setHasHatMatrix(true);
    REQUIRE_NOTHROW(algorithm.run());

    VariablesCriterionList criterions = algorithm.indepVarCriterionList();
    REQUIRE_THAT(variables2indices(criterions[0].first), Catch::Equals(vector<int>({ 2 })));
    REQUIRE_THAT(criterions[0].second, Catch::WithinAbs(2567.715486436010, 1e-8));
    REQUIRE_THAT(variables2indices(criterions[1].first), Catch::Equals(vector<int>({ 3 })));
    REQUIRE_THAT(criterions[1].second, Catch::WithinAbs(2566.326946530555, 1e-8));
    REQUIRE_THAT(variables2indices(criterions[2].first), Catch::Equals(vector<int>({ 1 })));
    REQUIRE_THAT(criterions[2].second, Catch::WithinAbs(2446.849320258895, 1e-8));
    REQUIRE_THAT(variables2indices(criterions[3].first), Catch::Equals(vector<int>({ 1, 3 })));
    REQUIRE_THAT(criterions[3].second, Catch::WithinAbs(2456.445607367164, 1e-8));
    REQUIRE_THAT(variables2indices(criterions[4].first), Catch::Equals(vector<int>({ 1, 2 })));
    REQUIRE_THAT(criterions[4].second, Catch::WithinAbs(2452.651290897472, 1e-8));
    REQUIRE_THAT(variables2indices(criterions[5].first), Catch::Equals(vector<int>({ 1, 2, 3 })));
    REQUIRE_THAT(criterions[5].second, Catch::WithinAbs(2465.692469262322, 1e-8));
}

TEST_CASE("GWDR: basic flow (multithread)")
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

    uword nDim = londonhp.points().n_cols;
    vector<CGwmSpatialWeight> spatials;
    for (size_t i = 0; i < nDim; i++)
    {
        CGwmOneDimDistance distance;
        CGwmBandwidthWeight bandwidth(0, true, CGwmBandwidthWeight::Bisquare);
        spatials.push_back(CGwmSpatialWeight(&bandwidth, &distance));
    }

    GwmVariable purchase(0, true, "PURCHASE");
    GwmVariable floorsz(1, true, "FLOORSZ");
    GwmVariable unemploy(2, true, "UNEMPLOY");
    GwmVariable prof(3, true, "PROF");
    vector<GwmVariable> indepVars = { floorsz, unemploy, prof };

    CGwmGWDR algorithm;
    algorithm.setSourceLayer(&londonhp);
    algorithm.setDependentVariable(purchase);
    algorithm.setIndependentVariables(indepVars);
    algorithm.setSpatialWeights(spatials);
    algorithm.setEnableBandwidthOptimize(true);
    algorithm.setEnableIndepVarSelect(true);
    algorithm.setBandwidthCriterionType(CGwmGWDR::AIC);
    algorithm.setHasHatMatrix(true);
    algorithm.setParallelType(ParallelType::OpenMP);
    REQUIRE_NOTHROW(algorithm.run());

    VariablesCriterionList criterions = algorithm.indepVarCriterionList();
    REQUIRE_THAT(variables2indices(criterions[0].first), Catch::Equals(vector<int>({ 2 })));
    REQUIRE_THAT(criterions[0].second, Catch::WithinAbs(2551.613590205991, 1e-8));
    REQUIRE_THAT(variables2indices(criterions[1].first), Catch::Equals(vector<int>({ 3 })));
    REQUIRE_THAT(criterions[1].second, Catch::WithinAbs(2551.300322013486, 1e-8));
    REQUIRE_THAT(variables2indices(criterions[2].first), Catch::Equals(vector<int>({ 1 })));
    REQUIRE_THAT(criterions[2].second, Catch::WithinAbs(2468.932362800127, 1e-8));
    REQUIRE_THAT(variables2indices(criterions[3].first), Catch::Equals(vector<int>({ 1, 3 })));
    REQUIRE_THAT(criterions[3].second, Catch::WithinAbs(2452.864479420329, 1e-8));
    REQUIRE_THAT(variables2indices(criterions[4].first), Catch::Equals(vector<int>({ 1, 2 })));
    REQUIRE_THAT(criterions[4].second, Catch::WithinAbs(2450.596426665086, 1e-8));
    REQUIRE_THAT(variables2indices(criterions[5].first), Catch::Equals(vector<int>({ 1, 2, 3 })));
    REQUIRE_THAT(criterions[5].second, Catch::WithinAbs(2452.803889346232, 1e-8));

    const vector<CGwmSpatialWeight>& spatialWeights = algorithm.spatialWeights();
    REQUIRE_THAT(spatialWeights[0].weight<CGwmBandwidthWeight>()->bandwidth(), Catch::WithinAbs(81, 1e-12));
    REQUIRE_THAT(spatialWeights[1].weight<CGwmBandwidthWeight>()->bandwidth(), Catch::WithinAbs(142, 1e-12));

    GwmRegressionDiagnostic diagnostic = algorithm.diagnostic();
    REQUIRE_THAT(diagnostic.AICc, Catch::WithinAbs(2443.390268119515, 1e-6));
    REQUIRE_THAT(diagnostic.RSquare, Catch::WithinAbs(0.7249850738779454, 1e-6));
    REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::WithinAbs(0.6912744287973307, 1e-6));
}

TEST_CASE("GWDR: basic flow with bandwidth optimization (CV, multithread)")
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

    uword nDim = londonhp.points().n_cols;
    vector<CGwmSpatialWeight> spatials;
    for (size_t i = 0; i < nDim; i++)
    {
        CGwmOneDimDistance distance;
        CGwmBandwidthWeight bandwidth(0, true, CGwmBandwidthWeight::Bisquare);
        spatials.push_back(CGwmSpatialWeight(&bandwidth, &distance));
    }

    GwmVariable purchase(0, true, "PURCHASE");
    GwmVariable floorsz(1, true, "FLOORSZ");
    GwmVariable unemploy(2, true, "UNEMPLOY");
    GwmVariable prof(3, true, "PROF");
    vector<GwmVariable> indepVars = { floorsz, unemploy, prof };

    CGwmGWDR algorithm;
    algorithm.setSourceLayer(&londonhp);
    algorithm.setDependentVariable(purchase);
    algorithm.setIndependentVariables(indepVars);
    algorithm.setSpatialWeights(spatials);
    algorithm.setEnableBandwidthOptimize(true);
    algorithm.setBandwidthOptimizeStep(0.01);
    algorithm.setBandwidthCriterionType(CGwmGWDR::CV);
    algorithm.setHasHatMatrix(true);
    algorithm.setParallelType(ParallelType::OpenMP);
    REQUIRE_NOTHROW(algorithm.run());

    const vector<CGwmSpatialWeight>& spatialWeights = algorithm.spatialWeights();
    REQUIRE_THAT(spatialWeights[0].weight<CGwmBandwidthWeight>()->bandwidth(), Catch::WithinAbs(80, 1e-12));
    REQUIRE_THAT(spatialWeights[1].weight<CGwmBandwidthWeight>()->bandwidth(), Catch::WithinAbs(2550816, 1e-12));
}
