#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>

#include <vector>
#include <string>
#include <armadillo>

#include "gwmodelpp/CGwmSimpleLayer.h"
#include "gwmodelpp/CGwmGTWR.h"
#include "gwmodelpp/spatialweight/CGwmCRSTDistance.h"
#include "gwmodelpp/spatialweight/CGwmBandwidthWeight.h"
#include "gwmodelpp/spatialweight/CGwmSpatialWeight.h"
#include "gwmodelpp/GwmVariable.h"

#include "include/londonhp100.h"

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

TEST_CASE("GTWR: basic flow")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100temporal(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100temporal data.");
    }

    CGwmSimpleLayer londonhp(londonhp100_coord, londonhp100_data, londonhp100_fields);
    REQUIRE(londonhp.points().n_rows);
    REQUIRE(londonhp.data().n_rows);
    REQUIRE(londonhp.fields().size());
    REQUIRE(londonhp.featureCount());

    double lambda=0.05;
    CGwmCRSTDistance distance(false,lambda);
    CGwmBandwidthWeight bandwidth(74, true, CGwmBandwidthWeight::Gaussian);
    CGwmSpatialWeight spatial(&bandwidth, &distance);

    GwmVariable purchase(0, true, "PURCHASE");
    GwmVariable floorsz(1, true, "FLOORSZ");
    GwmVariable unemploy(2, true, "UNEMPLOY");
    GwmVariable prof(3, true, "PROF");
    vector<GwmVariable> indepVars = { floorsz, unemploy, prof };

    CGwmGTWR algorithm;
    algorithm.setSourceLayer(&londonhp);
    algorithm.setDependentVariable(purchase);
    algorithm.setIndependentVariables(indepVars);
    algorithm.setSpatialWeight(spatial);
    algorithm.setHasHatMatrix(true);
    REQUIRE_NOTHROW(algorithm.run());

    GwmRegressionDiagnostic diagnostic = algorithm.diagnostic();
    REQUIRE(abs(diagnostic.AIC - 2442.4601876846 ) < 1e-8);
    REQUIRE(abs(diagnostic.AICc - 2451.0253947922 ) < 1e-8);
    REQUIRE(abs(diagnostic.RSquare - 0.68332416707988 ) < 1e-8);
    REQUIRE(abs(diagnostic.RSquareAdjust - 0.66065045315908 ) < 1e-8);
}
