#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>

#include <vector>
#include <string>
#include <armadillo>
#include "gwmodelpp/CGwmMGWR.h"

#include "gwmodelpp/spatialweight/CGwmCRSDistance.h"
#include "gwmodelpp/spatialweight/CGwmBandwidthWeight.h"
#include "gwmodelpp/spatialweight/CGwmSpatialWeight.h"
#include "londonhp100.h"

using namespace std;
using namespace arma;


TEST_CASE("MGWR: basic flow")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    uword nVar = 3;
    vector<CGwmSpatialWeight> spatials;
    vector<bool> preditorCentered;
    vector<CGwmMGWR::BandwidthInitilizeType> bandwidthInitialize;
    vector<CGwmMGWR::BandwidthSelectionCriterionType> bandwidthSelectionApproach;
    for (size_t i = 0; i < nVar; i++)
    {
        CGwmCRSDistance distance;
        CGwmBandwidthWeight bandwidth(0, false, CGwmBandwidthWeight::Bisquare);
        spatials.push_back(CGwmSpatialWeight(&bandwidth, &distance));
        preditorCentered.push_back(i != 0);
        bandwidthInitialize.push_back(CGwmMGWR::BandwidthInitilizeType::Null);
        bandwidthSelectionApproach.push_back(CGwmMGWR::BandwidthSelectionCriterionType::CV);
    }

    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_data.n_rows), londonhp100_data.cols(1, 3));

    CGwmMGWR algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeights(spatials);
    algorithm.setHasHatMatrix(true);
    algorithm.setPreditorCentered(preditorCentered);
    algorithm.setBandwidthInitilize(bandwidthInitialize);
    algorithm.setBandwidthSelectionApproach(bandwidthSelectionApproach);
    algorithm.setBandwidthSelectThreshold(vector(3, 1e-5));
    REQUIRE_NOTHROW(algorithm.fit());

    const vector<CGwmSpatialWeight>& spatialWeights = algorithm.spatialWeights();
    REQUIRE_THAT(spatialWeights[0].weight<CGwmBandwidthWeight>()->bandwidth(), Catch::WithinAbs(4623.78, 0.1));
    REQUIRE_THAT(spatialWeights[1].weight<CGwmBandwidthWeight>()->bandwidth(), Catch::WithinAbs(12665.70, 0.1));
    REQUIRE_THAT(spatialWeights[2].weight<CGwmBandwidthWeight>()->bandwidth(), Catch::WithinAbs(12665.70, 0.1));

    GwmRegressionDiagnostic diagnostic = algorithm.diagnostic();
    REQUIRE_THAT(diagnostic.AICc, Catch::WithinAbs(2437.09277417389, 1e-6));
    REQUIRE_THAT(diagnostic.RSquare, Catch::WithinAbs(0.744649364494, 1e-6));
    REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::WithinAbs(0.712344894394, 1e-6));
}

TEST_CASE("MGWR: adaptive bandwidth autoselection of with AIC")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    uword nVar = 3;
    vector<CGwmSpatialWeight> spatials;
    vector<bool> preditorCentered;
    vector<CGwmMGWR::BandwidthInitilizeType> bandwidthInitialize;
    vector<CGwmMGWR::BandwidthSelectionCriterionType> bandwidthSelectionApproach;
    for (size_t i = 0; i < nVar; i++)
    {
        CGwmCRSDistance distance;
        CGwmBandwidthWeight bandwidth(36, true, CGwmBandwidthWeight::Bisquare);
        spatials.push_back(CGwmSpatialWeight(&bandwidth, &distance));
        preditorCentered.push_back(i != 0);
        bandwidthInitialize.push_back(CGwmMGWR::BandwidthInitilizeType::Initial);
        bandwidthSelectionApproach.push_back(CGwmMGWR::BandwidthSelectionCriterionType::AIC);
    }

    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_data.n_rows), londonhp100_data.cols(1, 3));

    CGwmMGWR algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeights(spatials);
    algorithm.setHasHatMatrix(true);
    algorithm.setCriterionType(CGwmMGWR::BackFittingCriterionType::dCVR);
    algorithm.setPreditorCentered(preditorCentered);
    algorithm.setBandwidthInitilize(bandwidthInitialize);
    algorithm.setBandwidthSelectionApproach(bandwidthSelectionApproach);
    algorithm.setBandwidthSelectRetryTimes(5);
    algorithm.setBandwidthSelectThreshold(vector(3, 1e-5));
    REQUIRE_NOTHROW(algorithm.fit());

    const vector<CGwmSpatialWeight>& spatialWeights = algorithm.spatialWeights();
    REQUIRE_THAT(spatialWeights[0].weight<CGwmBandwidthWeight>()->bandwidth(), Catch::WithinAbs(45, 0.1));
    REQUIRE_THAT(spatialWeights[1].weight<CGwmBandwidthWeight>()->bandwidth(), Catch::WithinAbs(98, 0.1));
    REQUIRE_THAT(spatialWeights[2].weight<CGwmBandwidthWeight>()->bandwidth(), Catch::WithinAbs(98, 0.1));

    GwmRegressionDiagnostic diagnostic = algorithm.diagnostic();
    REQUIRE_THAT(diagnostic.AICc, Catch::WithinAbs(2437.935218705351, 1e-6));
    REQUIRE_THAT(diagnostic.RSquare, Catch::WithinAbs(0.7486787930045755, 1e-6));
    REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::WithinAbs(0.7118919517893492, 1e-6));
}


TEST_CASE("MGWR: adaptive bandwidth autoselection of with CV")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    uword nVar = 3;
    vector<CGwmSpatialWeight> spatials;
    vector<bool> preditorCentered;
    vector<CGwmMGWR::BandwidthInitilizeType> bandwidthInitialize;
    vector<CGwmMGWR::BandwidthSelectionCriterionType> bandwidthSelectionApproach;
    for (size_t i = 0; i < nVar; i++)
    {
        CGwmCRSDistance distance;
        CGwmBandwidthWeight bandwidth(0, true, CGwmBandwidthWeight::Bisquare);
        spatials.push_back(CGwmSpatialWeight(&bandwidth, &distance));
        preditorCentered.push_back(i != 0);
        bandwidthInitialize.push_back(CGwmMGWR::BandwidthInitilizeType::Null);
        bandwidthSelectionApproach.push_back(CGwmMGWR::BandwidthSelectionCriterionType::CV);
    }

    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_data.n_rows), londonhp100_data.cols(1, 3));

    CGwmMGWR algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeights(spatials);
    algorithm.setHasHatMatrix(true);
    algorithm.setCriterionType(CGwmMGWR::BackFittingCriterionType::dCVR);
    algorithm.setPreditorCentered(preditorCentered);
    algorithm.setBandwidthInitilize(bandwidthInitialize);
    algorithm.setBandwidthSelectionApproach(bandwidthSelectionApproach);
    algorithm.setBandwidthSelectRetryTimes(5);
    algorithm.setBandwidthSelectThreshold(vector(3, 1e-5));
    REQUIRE_NOTHROW(algorithm.fit());

    const vector<CGwmSpatialWeight>& spatialWeights = algorithm.spatialWeights();
    REQUIRE_THAT(spatialWeights[0].weight<CGwmBandwidthWeight>()->bandwidth(), Catch::WithinAbs(35, 0.1));
    REQUIRE_THAT(spatialWeights[1].weight<CGwmBandwidthWeight>()->bandwidth(), Catch::WithinAbs(98, 0.1));
    REQUIRE_THAT(spatialWeights[2].weight<CGwmBandwidthWeight>()->bandwidth(), Catch::WithinAbs(98, 0.1));

    GwmRegressionDiagnostic diagnostic = algorithm.diagnostic();
    REQUIRE_THAT(diagnostic.AICc, Catch::WithinAbs(2438.256543499568, 1e-6));
    REQUIRE_THAT(diagnostic.RSquare, Catch::WithinAbs(0.757377391648, 1e-6));
    REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::WithinAbs(0.715598248202, 1e-6));
}

TEST_CASE("MGWR: basic flow with CVR")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    uword nVar = 3;
    vector<CGwmSpatialWeight> spatials;
    vector<bool> preditorCentered;
    vector<CGwmMGWR::BandwidthInitilizeType> bandwidthInitialize;
    vector<CGwmMGWR::BandwidthSelectionCriterionType> bandwidthSelectionApproach;
    for (size_t i = 0; i < nVar; i++)
    {
        CGwmCRSDistance distance;
        CGwmBandwidthWeight bandwidth(36, false, CGwmBandwidthWeight::Bisquare);
        spatials.push_back(CGwmSpatialWeight(&bandwidth, &distance));
        preditorCentered.push_back(i != 0);
        bandwidthInitialize.push_back(CGwmMGWR::BandwidthInitilizeType::Initial);
        bandwidthSelectionApproach.push_back(CGwmMGWR::BandwidthSelectionCriterionType::CV);
    }

    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_data.n_rows), londonhp100_data.cols(1, 3));

    CGwmMGWR algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeights(spatials);
    algorithm.setHasHatMatrix(true);
    algorithm.setCriterionType(CGwmMGWR::BackFittingCriterionType::CVR);
    algorithm.setPreditorCentered(preditorCentered);
    algorithm.setBandwidthInitilize(bandwidthInitialize);
    algorithm.setBandwidthSelectionApproach(bandwidthSelectionApproach);
    algorithm.setBandwidthSelectRetryTimes(5);
    algorithm.setBandwidthSelectThreshold(vector(3, 1e-5));
    algorithm.setParallelType(ParallelType::OpenMP);
    REQUIRE_NOTHROW(algorithm.fit());

    const vector<CGwmSpatialWeight>& spatialWeights = algorithm.spatialWeights();
    REQUIRE_THAT(spatialWeights[0].weight<CGwmBandwidthWeight>()->bandwidth(), Catch::WithinAbs(4623.78, 0.1));
    REQUIRE_THAT(spatialWeights[1].weight<CGwmBandwidthWeight>()->bandwidth(), Catch::WithinAbs(12665.70, 0.1));
    REQUIRE_THAT(spatialWeights[2].weight<CGwmBandwidthWeight>()->bandwidth(), Catch::WithinAbs(12665.70, 0.1));

    GwmRegressionDiagnostic diagnostic = algorithm.diagnostic();
    REQUIRE_THAT(diagnostic.AICc, Catch::WithinAbs(2437.09277417389, 1e-6));
    REQUIRE_THAT(diagnostic.RSquare, Catch::WithinAbs(0.744649364494, 1e-6));
    REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::WithinAbs(0.712344894394, 1e-6));
}


TEST_CASE("MGWR: basic flow (multithread)")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    uword nVar = 3;
    vector<CGwmSpatialWeight> spatials;
    vector<bool> preditorCentered;
    vector<CGwmMGWR::BandwidthInitilizeType> bandwidthInitialize;
    vector<CGwmMGWR::BandwidthSelectionCriterionType> bandwidthSelectionApproach;
    for (size_t i = 0; i < nVar; i++)
    {
        CGwmCRSDistance distance;
        CGwmBandwidthWeight bandwidth(0, true, CGwmBandwidthWeight::Bisquare);
        spatials.push_back(CGwmSpatialWeight(&bandwidth, &distance));
        preditorCentered.push_back(i != 0);
        bandwidthInitialize.push_back(CGwmMGWR::BandwidthInitilizeType::Null);
        bandwidthSelectionApproach.push_back(CGwmMGWR::BandwidthSelectionCriterionType::CV);
    }

    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_data.n_rows), londonhp100_data.cols(1, 3));

    CGwmMGWR algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeights(spatials);
    algorithm.setHasHatMatrix(true);
    algorithm.setCriterionType(CGwmMGWR::BackFittingCriterionType::dCVR);
    algorithm.setPreditorCentered(preditorCentered);
    algorithm.setBandwidthInitilize(bandwidthInitialize);
    algorithm.setBandwidthSelectionApproach(bandwidthSelectionApproach);
    algorithm.setBandwidthSelectRetryTimes(5);
    algorithm.setBandwidthSelectThreshold(vector(3, 1e-5));
    algorithm.setParallelType(ParallelType::OpenMP);
    REQUIRE_NOTHROW(algorithm.fit());

    const vector<CGwmSpatialWeight>& spatialWeights = algorithm.spatialWeights();
    REQUIRE_THAT(spatialWeights[0].weight<CGwmBandwidthWeight>()->bandwidth(), Catch::WithinAbs(35, 0.1));
    REQUIRE_THAT(spatialWeights[1].weight<CGwmBandwidthWeight>()->bandwidth(), Catch::WithinAbs(98, 0.1));
    REQUIRE_THAT(spatialWeights[2].weight<CGwmBandwidthWeight>()->bandwidth(), Catch::WithinAbs(98, 0.1));

    GwmRegressionDiagnostic diagnostic = algorithm.diagnostic();
    REQUIRE_THAT(diagnostic.AICc, Catch::WithinAbs(2438.256543499568, 1e-6));
    REQUIRE_THAT(diagnostic.RSquare, Catch::WithinAbs(0.757377391648, 1e-6));
    REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::WithinAbs(0.715598248202, 1e-6));
}

