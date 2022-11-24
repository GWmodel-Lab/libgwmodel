#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <vector>
#include <string>
#include <armadillo>
#include "gwmodelpp/CGwmGWPCA.h"
#include "gwmodelpp/spatialweight/CGwmCRSDistance.h"
#include "gwmodelpp/spatialweight/CGwmBandwidthWeight.h"
#include "gwmodelpp/spatialweight/CGwmSpatialWeight.h"
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

    CGwmCRSDistance distance(false);
    CGwmBandwidthWeight bandwidth(36, true, CGwmBandwidthWeight::Gaussian);
    CGwmSpatialWeight spatial(&bandwidth, &distance);

    mat x = londonhp100_data.cols(1, 3);

    CGwmGWPCA algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setVariables(x);
    algorithm.setSpatialWeight(spatial);
    algorithm.setKeepComponents(2);
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
        { 0.997738665169, -0.01152923886484, -0.0404508300357 },
        { 0.998673840690, -0.00822122467004, -0.0046831832351 },
        { 0.999297415085, -0.00389424492786,  0.0320948265474  },
        { 0.999678999647,  0.00274831974093,  0.0508510246498  },
        { 0.999999194544,  0.01053269924131,  0.0662213367046  }
    };
    mat loadings_pc1 = loadings.slice(0);
    vec loadings_pc1_sign = sign(loadings_pc1.col(0));
    loadings_pc1.each_col([&loadings_pc1_sign](colvec& c) { c %= loadings_pc1_sign; });
    mat loadings_pc1_q = quantile(loadings_pc1, p, 0);
    REQUIRE(approx_equal(loadings_pc1_q, loadings_pc1_q0, "absdiff", 1e-8));

    mat loadings_pc2_q0 = {
        { 6.28417560614e-05, -0.215135019168, -0.980384688419 },
        { 2.52111111610e-02, -0.204691596452, -0.976874091165 },
        { 3.74011742355e-02,  0.203737057043, -0.975636775071 },
        { 5.13759501838e-02,  0.214783181352,  0.976316483099 },
        { 6.71714511032e-02,  0.219162658504,  0.979248777221 }
    };
    mat loadings_pc2 = loadings.slice(1);
    vec loadings_pc2_sign = sign(loadings_pc2.col(0));
    loadings_pc2.each_col([&loadings_pc2_sign](colvec& c) { c %= loadings_pc2_sign; });
    mat loadings_pc2_q = quantile(loadings_pc2, p, 0);
    REQUIRE(approx_equal(loadings_pc2_q, loadings_pc2_q0, "absdiff", 1e-8));
}
