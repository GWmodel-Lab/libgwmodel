#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <vector>
#include <string>
#include <armadillo>
#include "gwmodelpp/CGwmGGWR.h"
#include "gwmodelpp/spatialweight/CGwmCRSDistance.h"
#include "gwmodelpp/spatialweight/CGwmBandwidthWeight.h"
#include "gwmodelpp/spatialweight/CGwmSpatialWeight.h"
#include "londonhp100.h"
#include "londonhp.h"

using namespace std;
using namespace arma;


TEST_CASE("GGWR: basic flow")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_coord.n_rows), londonhp100_data.cols(1, 3));

    CGwmCRSDistance distance(false);
    CGwmBandwidthWeight bandwidth(27, true, CGwmBandwidthWeight::Gaussian);
    CGwmSpatialWeight spatial(&bandwidth, &distance);

    CGwmGGWR algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeight(spatial);
    algorithm.setHasHatMatrix(true);
    REQUIRE_NOTHROW(algorithm.fit());

    GwmGGWRDiagnostic diagnostic = algorithm.getDiagnostic();
    REQUIRE_THAT(diagnostic.RSS, Catch::WithinAbs(942063.05166298, 1e-5));
    REQUIRE_THAT(diagnostic.AIC, Catch::WithinAbs(942081.2250579, 1e-5));
    REQUIRE_THAT(diagnostic.AICc, Catch::WithinAbs(942083.26379446, 1e-5));
    REQUIRE_THAT(diagnostic.RSquare, Catch::WithinAbs(0.72696698, 1e-5));
    
}

TEST_CASE("GGWR: adaptive bandwidth autoselection of with AIC")
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

    CGwmGGWR algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeight(spatial);
    algorithm.setHasHatMatrix(true);

    algorithm.setIsAutoselectBandwidth(true);
    algorithm.setBandwidthSelectionCriterionType(CGwmGGWR::BandwidthSelectionCriterionType::AIC);
    
    REQUIRE_NOTHROW(algorithm.fit());

    size_t bw = (size_t)algorithm.spatialWeight().weight<CGwmBandwidthWeight>()->bandwidth();
    REQUIRE(bw == 21);

    GwmGGWRDiagnostic diagnostic = algorithm.getDiagnostic();
    REQUIRE_THAT(diagnostic.RSS, Catch::WithinAbs(893682.63762606, 1e-5));
    REQUIRE_THAT(diagnostic.AIC, Catch::WithinAbs(893704.45041084, 1e-5));
    REQUIRE_THAT(diagnostic.AICc, Catch::WithinAbs(893707.39854274, 1e-5));
    REQUIRE_THAT(diagnostic.RSquare, Catch::WithinAbs(0.740988819, 1e-5));
    
}

TEST_CASE("GGWR: adaptive bandwidth autoselection and with binomial")
{
    mat londonhp_coord, londonhp_data;
    vector<string> londonhp_fields;
    if (!read_londonhp(londonhp_coord, londonhp_data, londonhp_fields))
    {
        FAIL("Cannot load londonhp data.");
    }

    CGwmCRSDistance distance(false);
    CGwmBandwidthWeight bandwidth(0, true, CGwmBandwidthWeight::Gaussian);
    CGwmSpatialWeight spatial(&bandwidth, &distance);

    vec y = londonhp_data.col(13);
    mat x = join_rows(ones(londonhp_coord.n_rows), londonhp_data.cols(1,1));

    CGwmGGWR algorithm;
    algorithm.setCoords(londonhp_coord);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeight(spatial);
    algorithm.setHasHatMatrix(true);
    algorithm.setFamily(CGwmGGWR::Family::Binomial);
    algorithm.setIsAutoselectBandwidth(true);
    
    REQUIRE_NOTHROW(algorithm.fit());

    size_t bw = (size_t)algorithm.spatialWeight().weight<CGwmBandwidthWeight>()->bandwidth();
    REQUIRE(bw == 275);
    
    REQUIRE_NOTHROW(algorithm.fit());

    GwmGGWRDiagnostic diagnostic = algorithm.getDiagnostic();
    REQUIRE_THAT(diagnostic.RSS, Catch::WithinAbs(149.496688455, 1e-5));
    REQUIRE_THAT(diagnostic.AIC, Catch::WithinAbs(154.56992046, 1e-5));
    REQUIRE_THAT(diagnostic.AICc, Catch::WithinAbs(154.62734183, 1e-5));
    REQUIRE_THAT(diagnostic.RSquare, Catch::WithinAbs(0.20968291, 1e-5));
  
}

#ifdef ENABLE_OPENMP
TEST_CASE("GGWR: multithread basic flow")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_coord.n_rows), londonhp100_data.cols(1, 3));

    CGwmCRSDistance distance(false);
    CGwmBandwidthWeight bandwidth(27, true, CGwmBandwidthWeight::Gaussian);
    CGwmSpatialWeight spatial(&bandwidth, &distance);

    CGwmGGWR algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeight(spatial);
    algorithm.setHasHatMatrix(true);
    algorithm.setIsAutoselectBandwidth(true);
    algorithm.setBandwidthSelectionCriterionType(CGwmGGWR::BandwidthSelectionCriterionType::CV);
    algorithm.setParallelType(ParallelType::OpenMP);
    algorithm.setOmpThreadNum(6);
    algorithm.setFamily(CGwmGGWR::Family::Poisson);
    REQUIRE_NOTHROW(algorithm.fit());

    double bw = algorithm.spatialWeight().weight<CGwmBandwidthWeight>()->bandwidth();
    REQUIRE(bw == 27.0);

    GwmGGWRDiagnostic diagnostic = algorithm.getDiagnostic();
    REQUIRE_THAT(diagnostic.RSS, Catch::WithinAbs(942063.05166298, 1e-5));
    REQUIRE_THAT(diagnostic.AIC, Catch::WithinAbs(942081.2250579, 1e-5));
    REQUIRE_THAT(diagnostic.AICc, Catch::WithinAbs(942083.26379446, 1e-5));
    REQUIRE_THAT(diagnostic.RSquare, Catch::WithinAbs(0.72696698, 1e-5));

}
#endif


