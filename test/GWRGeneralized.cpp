#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <vector>
#include <string>
#include <armadillo>
#include "gwmodelpp/GWRGeneralized.h"
#include "gwmodelpp/spatialweight/CRSDistance.h"
#include "gwmodelpp/spatialweight/BandwidthWeight.h"
#include "gwmodelpp/spatialweight/SpatialWeight.h"
#include "londonhp100.h"
#include "londonhp.h"
#include "TerminateCheckTelegram.h"

using namespace std;
using namespace arma;
using namespace gwm;


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

    CRSDistance distance(false);
    BandwidthWeight bandwidth(27, true, BandwidthWeight::Gaussian);
    SpatialWeight spatial(&bandwidth, &distance);

    GWRGeneralized algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeight(spatial);
    algorithm.setHasHatMatrix(true);
    REQUIRE_NOTHROW(algorithm.fit());

    GWRGeneralizedDiagnostic diagnostic = algorithm.getDiagnostic();
    /*REQUIRE_THAT(diagnostic.RSS, Catch::Matchers::WithinAbs(942063.05166298, 1e-5));
    REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(942081.2250579, 1e-5));
    REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(942083.26379446, 1e-5));
    REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.72696698, 1e-5));*/
    
}

TEST_CASE("GGWR: adaptive bandwidth autoselection of with AIC")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    CRSDistance distance(false);
    BandwidthWeight bandwidth(0, true, BandwidthWeight::Gaussian);
    SpatialWeight spatial(&bandwidth, &distance);

    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_coord.n_rows), londonhp100_data.cols(1, 3));

    GWRGeneralized algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeight(spatial);
    algorithm.setHasHatMatrix(true);

    algorithm.setIsAutoselectBandwidth(true);
    algorithm.setBandwidthSelectionCriterionType(GWRGeneralized::BandwidthSelectionCriterionType::AIC);
    
    REQUIRE_NOTHROW(algorithm.fit());

    size_t bw = (size_t)algorithm.spatialWeight().weight<BandwidthWeight>()->bandwidth();
    REQUIRE(bw == 21);

    GWRGeneralizedDiagnostic diagnostic = algorithm.getDiagnostic();
    /*REQUIRE_THAT(diagnostic.RSS, Catch::Matchers::WithinAbs(893682.63762606, 1e-5));
    REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(893704.45041084, 1e-5));
    REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(893707.39854274, 1e-5));
    REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.740988819, 1e-5));*/
    
}

TEST_CASE("GGWR: adaptive bandwidth autoselection and with binomial")
{
    mat londonhp_coord, londonhp_data;
    vector<string> londonhp_fields;
    if (!read_londonhp(londonhp_coord, londonhp_data, londonhp_fields))
    {
        FAIL("Cannot load londonhp data.");
    }

    CRSDistance distance(false);
    BandwidthWeight bandwidth(0, true, BandwidthWeight::Gaussian);
    SpatialWeight spatial(&bandwidth, &distance);

    vec y = londonhp_data.col(13);
    mat x = join_rows(ones(londonhp_coord.n_rows), londonhp_data.cols(1,1));

    GWRGeneralized algorithm;
    algorithm.setCoords(londonhp_coord);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeight(spatial);
    algorithm.setHasHatMatrix(true);
    algorithm.setFamily(GWRGeneralized::Family::Binomial);
    algorithm.setIsAutoselectBandwidth(true);
    
    REQUIRE_NOTHROW(algorithm.fit());

    size_t bw = (size_t)algorithm.spatialWeight().weight<BandwidthWeight>()->bandwidth();
    REQUIRE(bw == 275);
    
    REQUIRE_NOTHROW(algorithm.fit());

    GWRGeneralizedDiagnostic diagnostic = algorithm.getDiagnostic();
    REQUIRE_THAT(diagnostic.RSS, Catch::Matchers::WithinAbs(149.496688455, 1e-5));
    REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(154.56992046, 1e-5));
    REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(154.62734183, 1e-5));
    REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.20968291, 1e-5));
  
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

    CRSDistance distance(false);
    BandwidthWeight bandwidth(27, true, BandwidthWeight::Gaussian);
    SpatialWeight spatial(&bandwidth, &distance);

    GWRGeneralized algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeight(spatial);
    algorithm.setHasHatMatrix(true);
    algorithm.setIsAutoselectBandwidth(true);
    algorithm.setBandwidthSelectionCriterionType(GWRGeneralized::BandwidthSelectionCriterionType::CV);
    algorithm.setParallelType(ParallelType::OpenMP);
    algorithm.setOmpThreadNum(6);
    algorithm.setFamily(GWRGeneralized::Family::Poisson);
    REQUIRE_NOTHROW(algorithm.fit());

    double bw = algorithm.spatialWeight().weight<BandwidthWeight>()->bandwidth();
    REQUIRE(bw == 27.0);

    GWRGeneralizedDiagnostic diagnostic = algorithm.getDiagnostic();
    /*REQUIRE_THAT(diagnostic.RSS, Catch::Matchers::WithinAbs(942063.05166298, 1e-5));
    REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(942081.2250579, 1e-5));
    REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(942083.26379446, 1e-5));
    REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.72696698, 1e-5));*/

}
#endif


const map<GWRGeneralized::BandwidthSelectionCriterionType, string> BandwidthCriterionDict = {
    make_pair(GWRGeneralized::BandwidthSelectionCriterionType::AIC, "AIC"),
    make_pair(GWRGeneralized::BandwidthSelectionCriterionType::CV, "CV")
};

const map<GWRGeneralized::Family, string> FamilyDict = {
    make_pair(GWRGeneralized::Family::Binomial, "Binomial"),
    make_pair(GWRGeneralized::Family::Poisson, "Poisson")
};


TEST_CASE("GGWR: cancel")
{
    auto family = GENERATE(GWRGeneralized::Family::Poisson, GWRGeneralized::Family::Binomial);
    auto parallel = GENERATE(
        ParallelType::SerialOnly
#ifdef ENABLE_OPENMP
        , ParallelType::OpenMP
#endif  // ENABLE_OPENMP
    );

    SECTION("fit")
    {
        auto bandwidthCriterion = GENERATE(GWRGeneralized::BandwidthSelectionCriterionType::AIC, GWRGeneralized::BandwidthSelectionCriterionType::CV);
        auto stage = GENERATE(as<std::string>{}, "bandwidthSize", "fit");
        auto progress = GENERATE(0, 10);
        INFO("Settings: " << BandwidthCriterionDict.at(bandwidthCriterion) << ", " << FamilyDict.at(family) << ", " << ParallelTypeDict.at(parallel) << ", " << stage << ", " << progress);

        mat londonhp_coord, londonhp_data;
        vector<string> londonhp_fields;
        vec y;
        mat x;
        
        switch (family)
        {
        case GWRGeneralized::Family::Binomial:
            if (!read_londonhp(londonhp_coord, londonhp_data, londonhp_fields))
            {
                FAIL("Cannot load londonhp data.");
            }
            y = londonhp_data.col(13);
            x = join_rows(ones(londonhp_coord.n_rows), londonhp_data.cols(1,1));
            break;
        case GWRGeneralized::Family::Poisson:
            if (!read_londonhp100(londonhp_coord, londonhp_data, londonhp_fields))
            {
                FAIL("Cannot load londonhp data.");
            }
            y = londonhp_data.col(0);
            x = join_rows(ones(londonhp_coord.n_rows), londonhp_data.cols(1, 3));
            break;
        default:
            FAIL("Unknown family");
            break;
        }

        CRSDistance distance(false);
        BandwidthWeight bandwidth(0, true, BandwidthWeight::Gaussian);
        SpatialWeight spatial(&bandwidth, &distance);

        TerminateCheckTelegram *telegram = new TerminateCheckTelegram(stage, progress);
        GWRGeneralized algorithm;
        algorithm.setTelegram(telegram);
        algorithm.setCoords(londonhp_coord);
        algorithm.setDependentVariable(y);
        algorithm.setIndependentVariables(x);
        algorithm.setSpatialWeight(spatial);
        algorithm.setHasHatMatrix(true);
        algorithm.setIsAutoselectBandwidth(true);
        algorithm.setBandwidthSelectionCriterionType(bandwidthCriterion);
        algorithm.setFamily(family);
        algorithm.setParallelType(parallel);
        REQUIRE_NOTHROW(algorithm.fit());
        REQUIRE(algorithm.status() == Status::Terminated);
    }

}
