#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <vector>
#include <string>
#include <armadillo>
#include "gwmodelpp/GWRRobust.h"
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

TEST_CASE("RobustGWR: Filtered")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_coord.n_rows), londonhp100_data.cols(1, 3));
/*
    //SimpleLayer londonhp(londonhp100_coord, londonhp100_data, londonhp100_fields);
    REQUIRE(londonhp.points().n_rows);
    REQUIRE(londonhp.data().n_rows);
    REQUIRE(londonhp.fields().size());
    REQUIRE(londonhp.featureCount());
*/
    CRSDistance distance(false);
    BandwidthWeight bandwidth(36, true, BandwidthWeight::Gaussian);
    SpatialWeight spatial(&bandwidth, &distance);
/*
    GwmVariable purchase(0, true, "PURCHASE");
    GwmVariable floorsz(1, true, "FLOORSZ");
    GwmVariable unemploy(2, true, "UNEMPLOY");
    GwmVariable prof(3, true, "PROF");
    vector<GwmVariable> indepVars = { floorsz, unemploy, prof };
*/
    GWRRobust algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeight(spatial);
    algorithm.setHasHatMatrix(true);
    algorithm.setFiltered(true);
    REQUIRE_NOTHROW(algorithm.fit());

   /*  RegressionDiagnostic diagnostic = algorithm.diagnostic();
    REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(2436.60445730413, 1e-8));
    REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2448.27206524754, 1e-8));
    REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.708010632044736, 1e-8));
    REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.674975341723766, 1e-8)); */
}

TEST_CASE("RobustGWR: noFiltered")
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
    BandwidthWeight bandwidth(36, true, BandwidthWeight::Gaussian);
    SpatialWeight spatial(&bandwidth, &distance);


    GWRRobust algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeight(spatial);
    algorithm.setHasHatMatrix(true);
    algorithm.setFiltered(false);
    REQUIRE_NOTHROW(algorithm.fit());

    /* RegressionDiagnostic diagnostic = algorithm.diagnostic();
    REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(2436.60445730413, 1e-8));
    REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2448.27206524754, 1e-8));
    REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.708010632044736, 1e-8));
    REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.674975341723766, 1e-8)); */
}

TEST_CASE("RobustGWR: adaptive bandwidth autoselection of with CV")
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
    BandwidthWeight bandwidth(0, true, BandwidthWeight::Gaussian);
    SpatialWeight spatial(&bandwidth, &distance);

    GWRRobust algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeight(spatial);
    algorithm.setHasHatMatrix(true);

    algorithm.setIsAutoselectBandwidth(true);
    algorithm.setBandwidthSelectionCriterion(GWRRobust::BandwidthSelectionCriterionType::CV);
    
    REQUIRE_NOTHROW(algorithm.fit());

    size_t bw = (size_t)algorithm.spatialWeight().weight<BandwidthWeight>()->bandwidth();
    REQUIRE(bw == 67);

    /* RegressionDiagnostic diagnostic = algorithm.diagnostic();
    REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(2441.232, 1e-3));
    REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2449.859, 1e-3));
    REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.6873385, 1e-6));
    REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.6643629, 1e-6)); */
}

TEST_CASE("RobustGWR: indepdent variable autoselection with AIC")
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
    BandwidthWeight bandwidth(36, true, BandwidthWeight::Gaussian);
    SpatialWeight spatial(&bandwidth, &distance);

    GWRRobust algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeight(spatial);
    algorithm.setHasHatMatrix(true);

    algorithm.setIsAutoselectIndepVars(true);
    algorithm.setIndepVarSelectionThreshold(3.0);
    
    REQUIRE_NOTHROW(algorithm.fit());

    VariablesCriterionList criterions = algorithm.indepVarsSelectionCriterionList();
    REQUIRE_THAT(criterions[0].first, Catch::Matchers::Equals(vector<size_t>({ 2 })));
    REQUIRE_THAT(criterions[0].second, Catch::Matchers::WithinAbs(2551.61359020599, 1e-8));
    REQUIRE_THAT(criterions[1].first, Catch::Matchers::Equals(vector<size_t>({ 3 })));
    REQUIRE_THAT(criterions[1].second, Catch::Matchers::WithinAbs(2551.30032201349, 1e-8));
    REQUIRE_THAT(criterions[2].first, Catch::Matchers::Equals(vector<size_t>({ 1 })));
    REQUIRE_THAT(criterions[2].second, Catch::Matchers::WithinAbs(2468.93236280013, 1e-8));
    REQUIRE_THAT(criterions[3].first, Catch::Matchers::Equals(vector<size_t>({ 1, 3 })));
    REQUIRE_THAT(criterions[3].second, Catch::Matchers::WithinAbs(2452.86447942033, 1e-8));
    REQUIRE_THAT(criterions[4].first, Catch::Matchers::Equals(vector<size_t>({ 1, 2 })));
    REQUIRE_THAT(criterions[4].second, Catch::Matchers::WithinAbs(2450.59642666509, 1e-8));
    REQUIRE_THAT(criterions[5].first, Catch::Matchers::Equals(vector<size_t>({ 1, 2, 3 })));
    REQUIRE_THAT(criterions[5].second, Catch::Matchers::WithinAbs(2452.80388934625, 1e-8));

    vector<size_t> selectedVariables = algorithm.selectedVariables();
    REQUIRE_THAT(selectedVariables, Catch::Matchers::Equals(vector<size_t>({1, 3})));

    /* RegressionDiagnostic diagnostic = algorithm.diagnostic();
    REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(2436.677, 1e-3));
    REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2445.703, 1e-3));
    REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.7021572, 1e-6));
    REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.6771693, 1e-6)); */
}

#ifdef ENABLE_OPENMP
TEST_CASE("RobustGWR: multithread basic flow")
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
    BandwidthWeight bandwidth(0, true, BandwidthWeight::Gaussian);
    SpatialWeight spatial(&bandwidth, &distance);

    GWRRobust algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeight(spatial);
    algorithm.setHasHatMatrix(true);
    algorithm.setIsAutoselectBandwidth(true);
    algorithm.setBandwidthSelectionCriterion(GWRBasic::BandwidthSelectionCriterionType::CV);
    algorithm.setIsAutoselectIndepVars(true);
    algorithm.setIndepVarSelectionThreshold(3.0);
    algorithm.setParallelType(ParallelType::OpenMP);
    algorithm.setOmpThreadNum(omp_get_num_threads());
    REQUIRE_NOTHROW(algorithm.fit());

    VariablesCriterionList criterions = algorithm.indepVarsSelectionCriterionList();
    REQUIRE_THAT(criterions[0].first, Catch::Matchers::Equals(vector<size_t>({ 2 })));
    REQUIRE_THAT(criterions[0].second, Catch::Matchers::WithinAbs(2551.61359020599, 1e-8));
    REQUIRE_THAT(criterions[1].first, Catch::Matchers::Equals(vector<size_t>({ 3 })));
    REQUIRE_THAT(criterions[1].second, Catch::Matchers::WithinAbs(2551.30032201349, 1e-8));
    REQUIRE_THAT(criterions[2].first, Catch::Matchers::Equals(vector<size_t>({ 1 })));
    REQUIRE_THAT(criterions[2].second, Catch::Matchers::WithinAbs(2468.93236280013, 1e-8));
    REQUIRE_THAT(criterions[3].first, Catch::Matchers::Equals(vector<size_t>({ 1, 3 })));
    REQUIRE_THAT(criterions[3].second, Catch::Matchers::WithinAbs(2452.86447942033, 1e-8));
    REQUIRE_THAT(criterions[4].first, Catch::Matchers::Equals(vector<size_t>({ 1, 2 })));
    REQUIRE_THAT(criterions[4].second, Catch::Matchers::WithinAbs(2450.59642666509, 1e-8));
    REQUIRE_THAT(criterions[5].first, Catch::Matchers::Equals(vector<size_t>({ 1, 2, 3 })));
    REQUIRE_THAT(criterions[5].second, Catch::Matchers::WithinAbs(2452.80388934625, 1e-8));

    vector<size_t> selectedVariables = algorithm.selectedVariables();
    REQUIRE_THAT(selectedVariables, Catch::Matchers::Equals(vector<size_t>({1, 3})));

    double bw = algorithm.spatialWeight().weight<BandwidthWeight>()->bandwidth();
    REQUIRE(bw == 31.0);

    /* RegressionDiagnostic diagnostic = algorithm.diagnostic();
    REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(2435.8161441795, 1e-8));
    REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2445.49629974057, 1e-8));
    REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.706143867720706, 1e-8));
    REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.678982114793865, 1e-8)); */
 
}
#endif

TEST_CASE("Robust GWR: cancel")
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
    BandwidthWeight bandwidth(0, true, BandwidthWeight::Gaussian);
    SpatialWeight spatial(&bandwidth, &distance);

    const initializer_list<ParallelType> parallel_list = {
        ParallelType::SerialOnly
#ifdef ENABLE_OPENMP
        , ParallelType::OpenMP
#endif // ENABLE_OPENMP     
    };
    auto parallel = GENERATE_REF(values(parallel_list));

    SECTION("fit")
    {
        auto bwCriterion = GENERATE(as<GWRRobust::BandwidthSelectionCriterionType>{}, 0, 1);
        auto stage = GENERATE(as<std::string>{}, "indepVars", "bandwidthSize", "fit");
        auto progress = GENERATE(0, 10);
        INFO("Settings: " << stage << ", " << progress);

        auto telegram = make_unique<TerminateCheckTelegram>(stage, progress);
        GWRRobust algorithm;
        algorithm.setTelegram(std::move(telegram));
        algorithm.setCoords(londonhp100_coord);
        algorithm.setDependentVariable(y);
        algorithm.setIndependentVariables(x);
        algorithm.setSpatialWeight(spatial);
        algorithm.setHasHatMatrix(true);
        algorithm.setIsAutoselectIndepVars(true);
        algorithm.setIndepVarSelectionThreshold(3.0);
        algorithm.setIsAutoselectBandwidth(true);
        algorithm.setBandwidthSelectionCriterion(bwCriterion);
        algorithm.setParallelType(parallel);
#ifdef ENABLE_OPENMP
        algorithm.setOmpThreadNum(omp_get_num_threads());
#endif // ENABLE_OPENMP
        REQUIRE_NOTHROW(algorithm.fit());
        REQUIRE(algorithm.status() == Status::Terminated);
    }

    SECTION("predict")
    {
        auto stage = GENERATE(as<std::string>{}, "predict");
        auto progress = GENERATE(0, 10);
        INFO("Settings: " << stage << ", " << progress);

        auto telegram = make_unique<TerminateCheckTelegram>(stage, progress);
        GWRRobust algorithm;
        algorithm.setTelegram(std::move(telegram));
        algorithm.setCoords(londonhp100_coord);
        algorithm.setDependentVariable(y);
        algorithm.setIndependentVariables(x);
        algorithm.setSpatialWeight(spatial);
        algorithm.setHasHatMatrix(true);
        algorithm.setIsAutoselectIndepVars(true);
        algorithm.setIndepVarSelectionThreshold(3.0);
        algorithm.setIsAutoselectBandwidth(true);
        algorithm.setBandwidthSelectionCriterion(GWRRobust::BandwidthSelectionCriterionType::AIC);
        algorithm.setParallelType(parallel);
#ifdef ENABLE_OPENMP
        algorithm.setOmpThreadNum(omp_get_num_threads());
#endif // ENABLE_OPENMP
        REQUIRE_NOTHROW(algorithm.fit());
        REQUIRE_NOTHROW(algorithm.predict(londonhp100_coord));
        REQUIRE(algorithm.status() == Status::Terminated);
    }
    
}
