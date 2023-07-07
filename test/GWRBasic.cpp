#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <vector>
#include <string>
#include <armadillo>
#include "gwmodelpp/GWRBasic.h"
#include "gwmodelpp/spatialweight/CRSDistance.h"
#include "gwmodelpp/spatialweight/BandwidthWeight.h"
#include "gwmodelpp/spatialweight/SpatialWeight.h"
#include "londonhp100.h"
#include "TerminateCheckTelegram.h"

using namespace std;
using namespace arma;
using namespace gwm;

TEST_CASE("BasicGWR: LondonHP")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }
    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_coord.n_rows), londonhp100_data.cols(1, 3));

    GWRBasic algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    CRSDistance distance(false);

    SECTION("adaptive bandwidth | no bandwidth optimization | no variable optimization | serial") {
        BandwidthWeight bandwidth(36, true, BandwidthWeight::Gaussian);
        SpatialWeight spatial(&bandwidth, &distance);
        algorithm.setSpatialWeight(spatial);
        REQUIRE_NOTHROW(algorithm.fit());
        REQUIRE(algorithm.hasIntercept() == true);
        RegressionDiagnostic diagnostic = algorithm.diagnostic();
        REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(2436.60445730413, 1e-8));
        REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2448.27206524754, 1e-8));
        REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.708010632044736, 1e-8));
        REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.674975341723766, 1e-8));
    }
    
    SECTION("fixed bandwidth | no bandwidth optimization | no variable optimization | serial") {
        BandwidthWeight bandwidth(5000, false, BandwidthWeight::Gaussian);
        SpatialWeight spatial(&bandwidth, &distance);
        algorithm.setSpatialWeight(spatial);
        REQUIRE_NOTHROW(algorithm.fit());
        RegressionDiagnostic diagnostic = algorithm.diagnostic();
        REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(2437.649574267587, 1e-8));
        REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2447.676281164379, 1e-8));
        REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.701466295457, 1e-8));
        REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.673691594464, 1e-8));
    }

    SECTION("adaptive bandwidth | CV bandwidth optimization | no variable optimization | serial") {
        BandwidthWeight bandwidth(0, true, BandwidthWeight::Gaussian);
        SpatialWeight spatial(&bandwidth, &distance);
        algorithm.setSpatialWeight(spatial);
        algorithm.setIsAutoselectBandwidth(true);
        algorithm.setBandwidthSelectionCriterion(GWRBasic::BandwidthSelectionCriterionType::CV);
        REQUIRE_NOTHROW(algorithm.fit());
        size_t bw = (size_t)algorithm.spatialWeight().weight<BandwidthWeight>()->bandwidth();
        REQUIRE(bw == 67);
    }
    
    SECTION("adaptive bandwidth | no bandwidth optimization | AIC variable optimization | serial") {
        BandwidthWeight bandwidth(36, true, BandwidthWeight::Gaussian);
        SpatialWeight spatial(&bandwidth, &distance);
        algorithm.setSpatialWeight(spatial);
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
    }

#ifdef ENABLE_OPENMP
    SECTION("adaptive bandwidth | CV bandwidth optimization | AIC variable optimization | omp parallel") {
        BandwidthWeight bandwidth(36, true, BandwidthWeight::Gaussian);
        SpatialWeight spatial(&bandwidth, &distance);
        algorithm.setSpatialWeight(spatial);
        algorithm.setIsAutoselectBandwidth(true);
        algorithm.setBandwidthSelectionCriterion(GWRBasic::BandwidthSelectionCriterionType::CV);
        algorithm.setIsAutoselectIndepVars(true);
        algorithm.setIndepVarSelectionThreshold(3.0);
        algorithm.setParallelType(ParallelType::OpenMP);
        algorithm.setOmpThreadNum(6);
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
        RegressionDiagnostic diagnostic = algorithm.diagnostic();
        REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(2435.8161441795, 1e-8));
        REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2445.49629974057, 1e-8));
        REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.706143867720706, 1e-8));
        REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.678982114793865, 1e-8));
    }
#endif

}


TEST_CASE("Basic GWR: cancel")
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

    vector<pair<string, size_t>> fit_stages = {
        make_pair("indepVarsSelection", 0),
        make_pair("indepVarsSelection", 10),
        make_pair("bandwidthSize", 0),
        make_pair("bandwidthSize", 10),
        make_pair("fit", 0),
        make_pair("fit", 10)
    };

    vector<pair<string, size_t>> predict_stages = {
        make_pair("predict", 0),
        make_pair("predict", 10)
    };

    SECTION("fit | CV Bandwidth | serial")
    {
        for (auto &&stage : fit_stages)
        {
            auto telegram = make_unique<TerminateCheckTelegram>(stage.first, stage.second);
            GWRBasic algorithm;
            algorithm.setTelegram(std::move(telegram));
            algorithm.setCoords(londonhp100_coord);
            algorithm.setDependentVariable(y);
            algorithm.setIndependentVariables(x);
            algorithm.setSpatialWeight(spatial);
            algorithm.setIsAutoselectBandwidth(true);
            algorithm.setIsAutoselectIndepVars(true);
            algorithm.setBandwidthSelectionCriterion(GWRBasic::BandwidthSelectionCriterionType::CV);
            REQUIRE_NOTHROW(algorithm.fit());
            REQUIRE(algorithm.status() == Status::Terminated);
        }
    }

    SECTION("fit | AIC Bandwidth | serial")
    {
        for (auto &&stage : fit_stages)
        {
            auto telegram = make_unique<TerminateCheckTelegram>(stage.first, stage.second);
            GWRBasic algorithm;
            algorithm.setTelegram(std::move(telegram));
            algorithm.setCoords(londonhp100_coord);
            algorithm.setDependentVariable(y);
            algorithm.setIndependentVariables(x);
            algorithm.setSpatialWeight(spatial);
            algorithm.setIsAutoselectBandwidth(true);
            algorithm.setIsAutoselectIndepVars(true);
            algorithm.setBandwidthSelectionCriterion(GWRBasic::BandwidthSelectionCriterionType::AIC);
            REQUIRE_NOTHROW(algorithm.fit());
            REQUIRE(algorithm.status() == Status::Terminated);
        }
    }

    SECTION("predict | serial")
    {
        for (auto &&stage : fit_stages)
        {
            auto telegram = make_unique<TerminateCheckTelegram>(stage.first, stage.second);
            GWRBasic algorithm;
            algorithm.setTelegram(std::move(telegram));
            algorithm.setCoords(londonhp100_coord);
            algorithm.setDependentVariable(y);
            algorithm.setIndependentVariables(x);
            algorithm.setSpatialWeight(spatial);
            algorithm.setIsAutoselectBandwidth(true);
            algorithm.setIsAutoselectIndepVars(true);
            algorithm.setBandwidthSelectionCriterion(GWRBasic::BandwidthSelectionCriterionType::AIC);
            REQUIRE_NOTHROW(algorithm.fit());
            REQUIRE_NOTHROW(algorithm.predict(londonhp100_coord));
            REQUIRE(algorithm.status() == Status::Terminated);
        }
    }

#ifdef ENABLE_OPENMP
    SECTION("fit | CV Bandwidth | openmp")
    {
        for (auto &&stage : fit_stages)
        {
            auto telegram = make_unique<TerminateCheckTelegram>(stage.first, stage.second);
            GWRBasic algorithm;
            algorithm.setTelegram(std::move(telegram));
            algorithm.setCoords(londonhp100_coord);
            algorithm.setDependentVariable(y);
            algorithm.setIndependentVariables(x);
            algorithm.setSpatialWeight(spatial);
            algorithm.setIsAutoselectBandwidth(true);
            algorithm.setIsAutoselectIndepVars(true);
            algorithm.setBandwidthSelectionCriterion(GWRBasic::BandwidthSelectionCriterionType::CV);
            algorithm.setParallelType(ParallelType::OpenMP);
            algorithm.setOmpThreadNum(6);
            REQUIRE_NOTHROW(algorithm.fit());
            REQUIRE(algorithm.status() == Status::Terminated);
        }
    }

    SECTION("fit | AIC Bandwidth | openmp")
    {
        for (auto &&stage : fit_stages)
        {
            auto telegram = make_unique<TerminateCheckTelegram>(stage.first, stage.second);
            GWRBasic algorithm;
            algorithm.setTelegram(std::move(telegram));
            algorithm.setCoords(londonhp100_coord);
            algorithm.setDependentVariable(y);
            algorithm.setIndependentVariables(x);
            algorithm.setSpatialWeight(spatial);
            algorithm.setIsAutoselectBandwidth(true);
            algorithm.setIsAutoselectIndepVars(true);
            algorithm.setBandwidthSelectionCriterion(GWRBasic::BandwidthSelectionCriterionType::AIC);
            algorithm.setParallelType(ParallelType::OpenMP);
            algorithm.setOmpThreadNum(6);
            REQUIRE_NOTHROW(algorithm.fit());
            REQUIRE(algorithm.status() == Status::Terminated);
        }
    }

    SECTION("predict | openmp")
    {
        for (auto &&stage : fit_stages)
        {
            auto telegram = make_unique<TerminateCheckTelegram>(stage.first, stage.second);
            GWRBasic algorithm;
            algorithm.setTelegram(std::move(telegram));
            algorithm.setCoords(londonhp100_coord);
            algorithm.setDependentVariable(y);
            algorithm.setIndependentVariables(x);
            algorithm.setSpatialWeight(spatial);
            algorithm.setIsAutoselectBandwidth(true);
            algorithm.setIsAutoselectIndepVars(true);
            algorithm.setBandwidthSelectionCriterion(GWRBasic::BandwidthSelectionCriterionType::AIC);
            algorithm.setParallelType(ParallelType::OpenMP);
            algorithm.setOmpThreadNum(6);
            REQUIRE_NOTHROW(algorithm.fit());
            REQUIRE_NOTHROW(algorithm.predict(londonhp100_coord));
            REQUIRE(algorithm.status() == Status::Terminated);
        }
    }
#endif  // ENABLE_OPENMP

}
