#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include <vector>
#include <armadillo>
#include "gwmodelpp/CGwmSimpleLayer.h"
#include "gwmodelpp/CGwmGWRBasic.h"
#include "gwmodelpp/spatialweight/CGwmCRSDistance.h"
#include "gwmodelpp/spatialweight/CGwmBandwidthWeight.h"
#include "gwmodelpp/spatialweight/CGwmSpatialWeight.h"
#include "gwmodelpp/GwmVariable.h"

using namespace std;
using namespace arma;

TEST_CASE("Check create algorithm")
{
    mat londonhp100_coord;
    field<std::string> coordHeader(2);
    coordHeader(0) = "x";
    coordHeader(1) = "y";
    REQUIRE(londonhp100_coord.load(arma::csv_name("../data/londonhp100coords.csv", coordHeader)));

    mat londonhp100_data;
    field<std::string> dataHeader(4);
    dataHeader(0) = "PURCHASE";
    dataHeader(1) = "FLOORSZ";
    dataHeader(2) = "UNEMPLOY";
    dataHeader(3) = "PROF";
    REQUIRE(londonhp100_data.load(arma::csv_name("../data/londonhp100data.csv", dataHeader)));

    vector<string> londonhp100_fields = 
    {
        "PURCHASE", "FLOORSZ", "UNEMPLOY", "PROF"
    };

    CGwmSimpleLayer* londonhp = new CGwmSimpleLayer(londonhp100_coord, londonhp100_data, londonhp100_fields);

    CGwmCRSDistance distance(false);
    CGwmBandwidthWeight bandwidth(36, true, CGwmBandwidthWeight::Gaussian);
    CGwmSpatialWeight spatial;
    spatial.setDistance(distance);
    spatial.setWeight(bandwidth);

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
}