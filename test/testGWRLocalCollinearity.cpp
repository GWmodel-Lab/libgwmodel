#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <vector>
#include <string>
#include <armadillo>
#include "gwmodelpp/GWRLocalCollinearity.h"
#include "gwmodelpp/spatialweight/CRSDistance.h"
#include "gwmodelpp/spatialweight/BandwidthWeight.h"
#include "gwmodelpp/spatialweight/SpatialWeight.h"
#include "londonhp100.h"
#include "TerminateCheckTelegram.h"

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif // ENABLE_OPENMP

using namespace std;
using namespace arma;
using namespace gwm;

TEST_CASE("LocalCollinearityGWR")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_coord.n_rows), londonhp100_data.cols(1, 3));

    SECTION("adaptive bandwidth | no bandwidth optimization | no lambda adjust")
    {

        CRSDistance distance(false);
        BandwidthWeight bandwidth(36, true, BandwidthWeight::Gaussian);
        SpatialWeight spatial(&bandwidth, &distance);

        GWRLocalCollinearity algorithm;
        algorithm.setCoords(londonhp100_coord);
        algorithm.setDependentVariable(y);
        algorithm.setIndependentVariables(x);
        algorithm.setSpatialWeight(spatial);
        algorithm.setHasHatMatrix(true);

        REQUIRE_NOTHROW(algorithm.fit());
        
        REQUIRE_THAT(algorithm.localCN().max(), Catch::Matchers::WithinAbs(60.433337574795, 1e-8));
        REQUIRE_THAT(algorithm.localCN().min(), Catch::Matchers::WithinAbs(42.800049204336, 1e-8));
        // REQUIRE_THAT(algorithm.localLambda().max(), Catch::Matchers::WithinAbs(0, 1e-8));

        RegressionDiagnostic diagnostic = algorithm.diagnostic();
        REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(2461.5654565, 1e-8));
        REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2464.60025589, 1e-8));
        REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.708010632043, 1e-8));
        REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.674975341722, 1e-8));

        REQUIRE(algorithm.hasIntercept() == true);
    }

    SECTION("adaptive bandwidth | bandwidth optimization CV | no lambda adjust")
    {
        CRSDistance distance(false);
        BandwidthWeight bandwidth(0, true, BandwidthWeight::Gaussian);
        SpatialWeight spatial(&bandwidth, &distance);

        GWRLocalCollinearity algorithm;
        algorithm.setCoords(londonhp100_coord);
        algorithm.setDependentVariable(y);
        algorithm.setIndependentVariables(x);
        algorithm.setSpatialWeight(spatial);
        algorithm.setHasHatMatrix(true);

        algorithm.setIsAutoselectBandwidth(true);
        algorithm.setBandwidthSelectionCriterion(GWRLocalCollinearity::BandwidthSelectionCriterionType::CV);

        REQUIRE_NOTHROW(algorithm.fit());

        size_t bw = (size_t)algorithm.spatialWeight().weight<BandwidthWeight>()->bandwidth();
        REQUIRE(bw == 67);
    }

    SECTION("adaptive bandwidth | no bandwidth optimization | lambda adjust | CnThresh=20 ")
    {

        CRSDistance distance(false);
        BandwidthWeight bandwidth(36, true, BandwidthWeight::Gaussian);
        SpatialWeight spatial(&bandwidth, &distance);

        GWRLocalCollinearity algorithm;
        algorithm.setCoords(londonhp100_coord);
        algorithm.setDependentVariable(y);
        algorithm.setIndependentVariables(x);
        algorithm.setSpatialWeight(spatial);
        algorithm.setHasHatMatrix(true);
        algorithm.setLambdaAdjust(true);
        // algorithm.setLambda(0);
        algorithm.setCnThresh(20);
        REQUIRE_NOTHROW(algorithm.fit());

        REQUIRE_THAT(algorithm.localCN().max(), Catch::Matchers::WithinAbs(60.433337574795, 1e-8));
        REQUIRE_THAT(algorithm.localCN().min(), Catch::Matchers::WithinAbs(42.800049204336, 1e-8));
        REQUIRE_THAT(algorithm.localLambda().max(), Catch::Matchers::WithinAbs(0.068751696228, 1e-8));
        REQUIRE_THAT(algorithm.localLambda().min(), Catch::Matchers::WithinAbs(0.054336147377, 1e-8));

        RegressionDiagnostic diagnostic = algorithm.diagnostic();
        REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(2461.8623182524, 1e-8));
        REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2464.8971176381, 1e-8));
        REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.70714253941241, 1e-8));
        REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.67400903424089, 1e-8));
    }

    SECTION("adaptive bandwidth | no bandwidth optimization | lambda 0.1 ")
    {

        CRSDistance distance(false);
        BandwidthWeight bandwidth(36, true, BandwidthWeight::Gaussian);
        SpatialWeight spatial(&bandwidth, &distance);

        GWRLocalCollinearity algorithm;
        algorithm.setCoords(londonhp100_coord);
        algorithm.setDependentVariable(y);
        algorithm.setIndependentVariables(x);
        algorithm.setSpatialWeight(spatial);
        algorithm.setHasHatMatrix(true);
        algorithm.setLambda(0.1);
        REQUIRE_NOTHROW(algorithm.fit());

        REQUIRE_THAT(algorithm.localCN().max(), Catch::Matchers::WithinAbs(60.433337574795, 1e-8));
        REQUIRE_THAT(algorithm.localCN().min(), Catch::Matchers::WithinAbs(42.800049204336, 1e-8));

        RegressionDiagnostic diagnostic = algorithm.diagnostic();
        REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(2462.0038025123, 1e-8));
        REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2465.0386018980, 1e-8));
        REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.706727898945, 1e-8));
        REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.673547481901, 1e-8));
    }
    
#ifdef ENABLE_OPENMP
    SECTION("adaptive bandwidth | bandwidth optimization ")
    {
        CRSDistance distance(false);
        BandwidthWeight bandwidth(0, true, BandwidthWeight::Gaussian);
        SpatialWeight spatial(&bandwidth, &distance);

        GWRLocalCollinearity algorithm;
        algorithm.setCoords(londonhp100_coord);
        algorithm.setDependentVariable(y);
        algorithm.setIndependentVariables(x);
        algorithm.setSpatialWeight(spatial);
        algorithm.setHasHatMatrix(true);
        algorithm.setIsAutoselectBandwidth(true);
        algorithm.setBandwidthSelectionCriterion(GWRLocalCollinearity::BandwidthSelectionCriterionType::CV);
        algorithm.setParallelType(ParallelType::OpenMP);
        algorithm.setOmpThreadNum(omp_get_num_threads());
        REQUIRE_NOTHROW(algorithm.fit());

        double bw = algorithm.spatialWeight().weight<BandwidthWeight>()->bandwidth();
        REQUIRE(bw == 67.0);

        REQUIRE_THAT(algorithm.localCN().max(), Catch::Matchers::WithinAbs(50.634464389348, 1e-8));
        REQUIRE_THAT(algorithm.localCN().min(), Catch::Matchers::WithinAbs(45.520575900277, 1e-8));

        RegressionDiagnostic diagnostic = algorithm.diagnostic();
        REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(2458.2472656218, 1e-8));
        REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2459.743757379, 1e-8));
        REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.6873384732363, 1e-8));
        REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.664362879709, 1e-8));
    }
#endif

}

TEST_CASE("LcGWR: cancel")
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

    const initializer_list<ParallelType> parallel_list = {
        ParallelType::SerialOnly
#ifdef ENABLE_OPENMP
        ,
        ParallelType::OpenMP
#endif // ENABLE_OPENMP
    };
    auto parallel = GENERATE_REF(values(parallel_list));

    SECTION("fit")
    {
        auto stage = GENERATE(as<std::string>{}, "bandwidthSize", "fit");
        auto progress = GENERATE(0, 10);
        INFO("Settings: "
             << "Parallel:" << parallel << ", Stage:" << stage << ", " << progress);

        auto telegram = make_unique<TerminateCheckTelegram>(stage, progress);
        GWRLocalCollinearity algorithm;
        algorithm.setTelegram(std::move(telegram));
        algorithm.setCoords(londonhp100_coord);
        algorithm.setDependentVariable(y);
        algorithm.setIndependentVariables(x);
        algorithm.setSpatialWeight(spatial);
        algorithm.setHasHatMatrix(true);
        algorithm.setIsAutoselectBandwidth(true);
        algorithm.setBandwidthSelectionCriterion(GWRLocalCollinearity::BandwidthSelectionCriterionType::CV);
        algorithm.setParallelType(parallel);
#ifdef ENABLE_OPENMP
        algorithm.setOmpThreadNum(omp_get_num_threads());
#endif // ENABLE_OPENMP
        REQUIRE_NOTHROW(algorithm.fit());
        REQUIRE(algorithm.status() == Status::Terminated);
    }
}
