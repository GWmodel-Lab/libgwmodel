#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <vector>
#include <string>
#include <armadillo>
#include "gwmodelpp/CGwmSimpleLayer.h"
#include "gwmodelpp/CGwmGWRBasic.h"
#include "gwmodelpp/spatialweight/CGwmCRSDistance.h"
#include "gwmodelpp/spatialweight/CGwmBandwidthWeight.h"
#include "gwmodelpp/spatialweight/CGwmSpatialWeight.h"
#include "gwmodelpp/GwmVariable.h"
#include "londonhp100.h"

using namespace std;
using namespace arma;

TEST_CASE("Basic Flow of BasicGWR")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    CGwmSimpleLayer* londonhp = new CGwmSimpleLayer(londonhp100_coord, londonhp100_data, londonhp100_fields);
    REQUIRE(londonhp->points().n_rows);
    REQUIRE(londonhp->data().n_rows);
    REQUIRE(londonhp->fields().size());
    REQUIRE(londonhp->featureCount());

    CGwmCRSDistance distance(false);
    CGwmBandwidthWeight bandwidth(36, true, CGwmBandwidthWeight::Gaussian);
    CGwmSpatialWeight spatial(&bandwidth, &distance);

    GwmVariable purchase = {0, true, "PURCHASE"};
    GwmVariable floorsz = {1, true, "FLOORSZ"};
    GwmVariable unemploy = {2, true, "UNEMPLOY"};
    GwmVariable prof = {3, true, "PROF"};
    vector<GwmVariable> indepVars = { floorsz, unemploy, prof };

    CGwmGWRBasic algorithm;
    algorithm.setSourceLayer(londonhp);
    algorithm.setDependentVariable(purchase);
    algorithm.setIndependentVariables(indepVars);
    algorithm.setSpatialWeight(spatial);
    algorithm.setHasHatMatrix(true);
    REQUIRE_NOTHROW(algorithm.run());

    GwmRegressionDiagnostic diagnostic = algorithm.diagnostic();
    REQUIRE(abs(diagnostic.AIC - 2436.60445730413) < 1e-8);
    REQUIRE(abs(diagnostic.AICc - 2448.27206524754) < 1e-8);
    REQUIRE(abs(diagnostic.RSquare - 0.708010632044736) < 1e-8);
    REQUIRE(abs(diagnostic.RSquareAdjust - 0.674975341723766) < 1e-8);
}