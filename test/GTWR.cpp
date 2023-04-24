#define CATCH_CONFIG_MAIN

#include <catch2/catch_all.hpp>

#include <vector>
#include <string>
#include <armadillo>


#include "gwmodelpp/spatialweight/CRSSTDistance.h"
#include "gwmodelpp/spatialweight/BandwidthWeight.h"
#include "gwmodelpp/spatialweight/SpatialWeight.h"
#include "gwmodelpp/GTWR.h"

#include "include/londonhp100.h"

using namespace std;
using namespace arma;
using namespace gwm;


TEST_CASE("GTWR: londonhp100")
{
    mat londonhp100_coord, londonhp100_data;
    vec londonhp100_times;
    vector<string> londonhp100_fields;
    if (!read_londonhp100temporal(londonhp100_coord, londonhp100_times, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100temporal data.");
    }

    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_coord.n_rows), londonhp100_data.cols(1, 3));

    GTWR algorithm;
    algorithm.setCoords(londonhp100_coord,londonhp100_times);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    CRSDistance sdist(false);
    OneDimDistance tdist;
    //lambda是空间距离的权重

    // SECTION(" lambda>1 | serial") {
    //     CRSSTDistance distance(&sdist, &tdist, 1.5);
    //     BandwidthWeight bandwidth(36,true, BandwidthWeight::Gaussian);
    //     SpatialWeight spatial(&bandwidth, &distance);
    //     // REQUIRE_THROWS(algorithm.fit());
    //     REQUIRE_THROWS(spatial);
    // }
    SECTION("adaptive bandwidth 36 | no bandwidth optimization | auto select lambda") {
        CRSSTDistance distance(&sdist, &tdist, 1);
        BandwidthWeight bandwidth(36,true, BandwidthWeight::Gaussian);
        SpatialWeight spatial(&bandwidth, &distance);
        algorithm.setSpatialWeight(spatial);
        algorithm.setHasHatMatrix(true);
        algorithm.setIsAutoselectLambda(true);
        // algorithm.getDistance(&distance);
        REQUIRE_NOTHROW(algorithm.fit());
        RegressionDiagnostic diagnostic = algorithm.diagnostic();
        REQUIRE(algorithm.hasIntercept() == true);
        ////lambda=0.8519019
        REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(2442.1827485575, 1e-8));
        REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2454.3397429928, 1e-8));
        REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.69230641334398, 1e-8));
        REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.65717667453514, 1e-8));
    }
    SECTION("adaptive bandwidth 36 | no bandwidth optimization | lambda=1 | angle=5/2*pi | serial") {
        CRSSTDistance distance(&sdist, &tdist, 1, 5 * arma::datum::pi / 2);
        BandwidthWeight bandwidth(36,true, BandwidthWeight::Gaussian);
        SpatialWeight spatial(&bandwidth, &distance);
        algorithm.setSpatialWeight(spatial);
        algorithm.setHasHatMatrix(true);
        REQUIRE_NOTHROW(algorithm.fit());
        RegressionDiagnostic diagnostic = algorithm.diagnostic();
        REQUIRE(algorithm.hasIntercept() == true);
        ////lambda=1
        REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(2436.6044573089, 1e-8));
        REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2448.2720652516, 1e-8));
        REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.7080106320292, 1e-8));
        REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.67497534170905, 1e-8));
    }
    SECTION("adaptive bandwidth 36 | no bandwidth optimization | lambda=0.05 | serial") {
        CRSSTDistance distance(&sdist, &tdist, 0.05);
        BandwidthWeight bandwidth(36,true, BandwidthWeight::Gaussian);
        SpatialWeight spatial(&bandwidth, &distance);
        algorithm.setSpatialWeight(spatial);
        algorithm.setHasHatMatrix(true);
        REQUIRE_NOTHROW(algorithm.fit());
        RegressionDiagnostic diagnostic = algorithm.diagnostic();
        REQUIRE(algorithm.hasIntercept() == true);
        //lambda=0.05
        REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(2442.2013929832, 1e-8));
        REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2454.3785650481, 1e-8));
        REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.69229184128813, 1e-8));
        REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.65696299229674, 1e-8));
    }
    SECTION("fixed bandwidth 5000 | no bandwidth optimization | lambda=1 | serial") {
        CRSSTDistance distance(&sdist, &tdist, 1);
        BandwidthWeight bandwidth(5000,false, BandwidthWeight::Gaussian);
        SpatialWeight spatial(&bandwidth, &distance);
        algorithm.setSpatialWeight(spatial);
        algorithm.setHasHatMatrix(true);
        REQUIRE_NOTHROW(algorithm.fit());
        RegressionDiagnostic diagnostic = algorithm.diagnostic();
        REQUIRE(algorithm.hasIntercept() == true);
        //lambda=1
        REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(2437.6495742714, 1e-8));
        REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2447.6762811677, 1e-8));
        REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.70146629544468, 1e-8));
        REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.67369159445212 , 1e-8));
    }
    SECTION("fixed bandwidth 5000 | no bandwidth optimization | lambda=0.05 | serial") {
        CRSSTDistance distance(&sdist, &tdist, 0.05);
        BandwidthWeight bandwidth(5000,false, BandwidthWeight::Gaussian);
        SpatialWeight spatial(&bandwidth, &distance);
        algorithm.setSpatialWeight(spatial);
        algorithm.setHasHatMatrix(true);
        REQUIRE_NOTHROW(algorithm.fit());
        RegressionDiagnostic diagnostic = algorithm.diagnostic();
        REQUIRE(algorithm.hasIntercept() == true);
        // lambda=0.05
        REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(2442.8360028579, 1e-8));
        REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2459.9274395074, 1e-8));
        REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.70005079437906, 1e-8));
        REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.66131126771988, 1e-8));
    }
    SECTION("fixed bandwidth | CV Gaussian bandwidth optimization | lambda=1 ") {
        CRSSTDistance distance(&sdist, &tdist, 1);
        BandwidthWeight bandwidth(0, false, BandwidthWeight::Gaussian);
        SpatialWeight spatial(&bandwidth, &distance);
        algorithm.setSpatialWeight(spatial);
        algorithm.setIsAutoselectBandwidth(true);
        algorithm.setBandwidthSelectionCriterion(GTWR::BandwidthSelectionCriterionType::CV);
        REQUIRE_NOTHROW(algorithm.fit());
        size_t bw = (size_t)algorithm.spatialWeight().weight<BandwidthWeight>()->bandwidth();
        REQUIRE(bw == 5076);
    }
    SECTION("adaptive bandwidth | AICc Bisquare bandwidth optimization | lambda=0.05 ") {
        CRSSTDistance distance(&sdist, &tdist, 0.05);
        BandwidthWeight bandwidth(0, true, BandwidthWeight::Bisquare);
        SpatialWeight spatial(&bandwidth, &distance);
        algorithm.setSpatialWeight(spatial);
        algorithm.setIsAutoselectBandwidth(true);
        algorithm.setBandwidthSelectionCriterion(GTWR::BandwidthSelectionCriterionType::AIC);
        REQUIRE_NOTHROW(algorithm.fit());
        size_t bw = (size_t)algorithm.spatialWeight().weight<BandwidthWeight>()->bandwidth();
        REQUIRE(bw == 70);
    }
    SECTION("adaptive bandwidth | CV bandwidth optimization | lambda=0.05 | omp parallel ") {
        CRSSTDistance distance(&sdist, &tdist, 0.05);
        BandwidthWeight bandwidth(0, true, BandwidthWeight::Gaussian);
        SpatialWeight spatial(&bandwidth, &distance);
        algorithm.setSpatialWeight(spatial);
        algorithm.setHasHatMatrix(true);
        algorithm.setIsAutoselectBandwidth(true);
        algorithm.setBandwidthSelectionCriterion(GTWR::BandwidthSelectionCriterionType::CV);
        algorithm.setParallelType(ParallelType::OpenMP);
        algorithm.setOmpThreadNum(6);
        REQUIRE_NOTHROW(algorithm.fit());
        RegressionDiagnostic diagnostic = algorithm.diagnostic();
        REQUIRE(algorithm.hasIntercept() == true);
        size_t bw = algorithm.spatialWeight().weight<BandwidthWeight>()->bandwidth();
        REQUIRE(bw == 46);
        REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(2443.5498915432, 1e-8));
        REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2453.2551390127, 1e-8));
        REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.6825745728157, 1e-8));
        REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.6551810065492, 1e-8));
    }
    SECTION("adaptive bandwidth | no bandwidth optimization | lambda=0.5 | omp parallel ") {
        CRSSTDistance distance(&sdist, &tdist, 0.5);
        BandwidthWeight bandwidth(46, true, BandwidthWeight::Gaussian);
        SpatialWeight spatial(&bandwidth, &distance);
        algorithm.setSpatialWeight(spatial);
        algorithm.setHasHatMatrix(true);
        algorithm.setParallelType(ParallelType::OpenMP);
        algorithm.setOmpThreadNum(6);
        REQUIRE_NOTHROW(algorithm.fit());
        RegressionDiagnostic diagnostic = algorithm.diagnostic();
        REQUIRE(algorithm.hasIntercept() == true);
        REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(2443.4855135383, 1e-8));
        REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2453.2073732101, 1e-8));
        REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.68281765798389, 1e-8));
        REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.65541987797358, 1e-8));
    }
}