#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <vector>
#include <string>
#include <armadillo>
#include "gwmodelpp/CGwmGWPCA.h"
#include "gwmodelpp/spatialweight/CGwmCRSDistance.h"
#include "gwmodelpp/spatialweight/CGwmBandwidthWeight.h"
#include "gwmodelpp/spatialweight/CGwmSpatialWeight.h"
#include "gwmodelpp/GwmVariable.h"
#include "londonhp100.h"

using namespace std;
using namespace arma;

TEST_CASE("GWPCA: basic flow")
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

    GwmVariable floorsz(1, true, "FLOORSZ");
    GwmVariable unemploy(2, true, "UNEMPLOY");
    GwmVariable prof(3, true, "PROF");
    vector<GwmVariable> variables = { floorsz, unemploy, prof };

    CGwmGWPCA algorithm;
    algorithm.setSourceLayer(londonhp);
    algorithm.setVariables(variables);
    algorithm.setSpatialWeight(spatial);
    REQUIRE_NOTHROW(algorithm.run());

    vec p = {0.0, 0.25, 0.5, 0.75, 1.0};

    mat comp_q0 = {
        { 86.09381920388,7.38948790899526 },
        { 87.2417310474256,10.0805823313445 },
        { 88.5114946422145,11.4166428700704 },
        { 89.8514496001622,12.6890545321313 },
        { 92.5449003124064,13.8382823156345 }
    };
    mat comp_q = quantile(algorithm.localPV(), p, 0);
    REQUIRE(approx_equal(comp_q, comp_q0, "absdiff", 1e-8));

    cube loadings = algorithm.loadings();

    mat loadings_pc1_q0 = {
        { 0.997738665168933,-0.0115292388648388,-0.040450830035712 },
        { 0.998673840689833,-0.00822122467004101,-0.00468318323510321 },
        { 0.999297415085303,-0.00389424492785976,0.0320948265474252 },
        { 0.999678999646886,0.00274831974093292,0.0508510246498129 },
        { 0.999999194544384,0.0105326992413149,0.0662213367046487 }
    };
    mat loadings_pc1_q = quantile(loadings.slice(0), p, 0);
    REQUIRE(approx_equal(loadings_pc1_q, loadings_pc1_q0, "absdiff", 1e-8));

    mat loadings_pc2_q0 = {
        { -0.0671714511032381,-0.219162658504117,-0.97924877722135 },
        { -0.0513759501838017,-0.214853304247932,0.976144875457391 },
        { -0.0321827960857794,-0.211329933955831,0.976665314794129 },
        {  0.00517581158157478,-0.204353440937033,0.978464099165948 },
        {  0.0417635544237787,0.202661857194208,0.980384688418526 }
    };
    mat loadings_pc2_q = quantile(loadings.slice(1), p, 0);
    REQUIRE(approx_equal(loadings_pc2_q, loadings_pc2_q0, "absdiff", 1e-8));
}
