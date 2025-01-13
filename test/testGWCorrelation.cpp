#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <vector>
#include <string>
#include <armadillo>
#include "gwmodelpp/GWCorrelation.h"
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

TEST_CASE("GWCorrelation: londonhp100")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    SECTION("adaptive bandwidth | GWCorrelation | serial")
    {
        mat x = londonhp100_data.cols(2, 3);
        mat y = londonhp100_data.cols(0, 1);
        uword nVar = 4;
        vector<SpatialWeight> spatials;
        vector<GWCorrelation::BandwidthInitilizeType> bandwidthInitialize;
        // vector<GWCorrelation::BandwidthSelectionCriterionType> bandwidthSelectionApproach;
        for (size_t i = 0; i < nVar; i++)
        {
            CRSDistance distance;
            BandwidthWeight bandwidth(36, true, BandwidthWeight::Gaussian);
            spatials.push_back(SpatialWeight(&bandwidth, &distance));
            bandwidthInitialize.push_back(GWCorrelation::BandwidthInitilizeType::Specified);
        }

        GWCorrelation algorithm;
        algorithm.setCoords(londonhp100_coord);
        algorithm.setVariables2(x);
        algorithm.setVariables1(y);
        algorithm.setSpatialWeights(spatials);
        algorithm.setBandwidthInitilize(bandwidthInitialize);
        // algorithm.setBandwidthSelectionApproach(bandwidthSelectionApproach);
        REQUIRE_NOTHROW(algorithm.run());

        vec p = {0.0, 0.25, 0.5, 0.75, 1.0};

        // mat localcov_q0 = {
        //     {-59963.99642, 180641.0455, -8.952112965, -34.03759576},
        //     {-55755.97844, 218271.5560, -5.884950904, -3.34033709},
        //     {-53872.53438, 243590.8673, -3.185232937, 26.98199512},
        //     {-46512.68133, 264445.5274, 2.355410295, 41.15969044},
        //     {-36165.71804, 288239.6796, 9.330908683, 52.39072918}};

        // mat localcov_q = quantile(algorithm.localCov(), p, 0);
        // REQUIRE(approx_equal(localcov_q, localcov_q0, "absdiff", 1e-1));

        mat localcorr_q0 = {
            {-0.3206001836, 0.2030111401, -0.12688297645, -0.08925682044},
            {-0.2968839387, 0.2466674205, -0.08536051649, -0.01004434385},
            {-0.2544518512, 0.2828306292, -0.04667167176, 0.08606678349},
            {-0.2409211543, 0.3241799401, 0.03032820334, 0.14221392650},
            {-0.2019074966, 0.3526574868, 0.10675955887, 0.16175140462}};

        mat localcorr_q = quantile(algorithm.localCorr(), p, 0);
        REQUIRE(approx_equal(localcorr_q, localcorr_q0, "absdiff", 1e-3));

        mat localscorr_q0 = {
            {-0.3865373154, 0.2720981003, -0.1329130573468, -0.070690496147},
            {-0.3675980290, 0.2823359549, -0.1003033138009, 0.003969910873},
            {-0.3338697101, 0.3360144608, -0.0756778419096, 0.073938787835},
            {-0.3145486110, 0.3580575086, -0.0002331551565, 0.108199984250},
            {-0.2965442864, 0.3807852261, 0.0690739762835, 0.170298974147}};

        mat localscorr_q = quantile(algorithm.localSCorr(), p, 0);
        localscorr_q.print();
        REQUIRE(approx_equal(localscorr_q, localscorr_q0, "absdiff", 1e-1));
    }

    SECTION("adaptive bandwidth | GWCorrelation | serial")
    {

        mat x = londonhp100_data.cols(1, 3);
        mat y = londonhp100_data.col(0);
        uword nVar = 3;

        vector<SpatialWeight> spatials;
        vector<GWCorrelation::BandwidthInitilizeType> bandwidthInitialize;
        vector<GWCorrelation::BandwidthSelectionCriterionType> bandwidthSelectionApproach;
        for (size_t i = 0; i < nVar; i++)
        {
            CRSDistance distance;
            BandwidthWeight bandwidth(36, true, BandwidthWeight::Bisquare);
            spatials.push_back(SpatialWeight(&bandwidth, &distance));
            bandwidthInitialize.push_back(GWCorrelation::BandwidthInitilizeType::Null);
            bandwidthSelectionApproach.push_back(GWCorrelation::BandwidthSelectionCriterionType::CV);
        }

        GWCorrelation algorithm;
        algorithm.setCoords(londonhp100_coord);
        algorithm.setVariables2(x);
        algorithm.setVariables1(y);
        algorithm.setSpatialWeights(spatials);
        algorithm.setBandwidthInitilize(bandwidthInitialize);
        algorithm.setBandwidthSelectionApproach(bandwidthSelectionApproach);
        REQUIRE_NOTHROW(algorithm.run());

        const vector<SpatialWeight>& spatialWeights = algorithm.spatialWeights();
        REQUIRE_THAT(spatialWeights[0].weight<BandwidthWeight>()->bandwidth(), Catch::Matchers::WithinAbs(68, 1e-3));
        REQUIRE_THAT(spatialWeights[1].weight<BandwidthWeight>()->bandwidth(), Catch::Matchers::WithinAbs(26, 1e-3));
        REQUIRE_THAT(spatialWeights[2].weight<BandwidthWeight>()->bandwidth(), Catch::Matchers::WithinAbs(98, 1e-3));

        vec p = {0.0, 0.25, 0.5, 0.75, 1.0};
        mat localcorr_q0 = {
            {0.5361748487, -0.3759779984, 0.1579405980},
            {0.7599386632, -0.2469136917, 0.2292532078},
            {0.7999516918, -0.1594374882, 0.2831406000},
            {0.8416192397, -0.1258046897, 0.3329573960},
            {0.8885400791,  0.2345363059, 0.3538947488}};
        mat localcorr_q = quantile(algorithm.localCorr(), p, 0);
        REQUIRE(approx_equal(localcorr_q, localcorr_q0, "absdiff", 1e-2));
    }

#ifdef ENABLE_OPENMP
    SECTION("adaptive bandwidth | GWCorrelation | omp parallel")
    {

        mat x = londonhp100_data.cols(1, 3);
        mat y = londonhp100_data.col(0);
        uword nVar = 3;

        vector<SpatialWeight> spatials;
        vector<GWCorrelation::BandwidthInitilizeType> bandwidthInitialize;
        vector<GWCorrelation::BandwidthSelectionCriterionType> bandwidthSelectionApproach;
        for (size_t i = 0; i < nVar; i++)
        {
            CRSDistance distance;
            BandwidthWeight bandwidth(36, true, BandwidthWeight::Bisquare);
            spatials.push_back(SpatialWeight(&bandwidth, &distance));
            bandwidthInitialize.push_back(GWCorrelation::BandwidthInitilizeType::Null);
            bandwidthSelectionApproach.push_back(GWCorrelation::BandwidthSelectionCriterionType::CV);
        }

        GWCorrelation algorithm;
        algorithm.setCoords(londonhp100_coord);
        algorithm.setVariables1(y);
        algorithm.setVariables2(x);
        algorithm.setSpatialWeights(spatials);
        algorithm.setBandwidthInitilize(bandwidthInitialize);
        algorithm.setBandwidthSelectionApproach(bandwidthSelectionApproach);
        algorithm.setParallelType(ParallelType::OpenMP);
        algorithm.setOmpThreadNum(omp_get_num_threads());
        REQUIRE_NOTHROW(algorithm.run());

        const vector<SpatialWeight>& spatialWeights = algorithm.spatialWeights();
        REQUIRE_THAT(spatialWeights[0].weight<BandwidthWeight>()->bandwidth(), Catch::Matchers::WithinAbs(68, 1e-3));
        REQUIRE_THAT(spatialWeights[1].weight<BandwidthWeight>()->bandwidth(), Catch::Matchers::WithinAbs(26, 1e-3));
        REQUIRE_THAT(spatialWeights[2].weight<BandwidthWeight>()->bandwidth(), Catch::Matchers::WithinAbs(98, 1e-3));

        vec p = {0.0, 0.25, 0.5, 0.75, 1.0};

        mat localcorr_q0 = {
            {0.5361748487, -0.3759779984, 0.1579405980},
            {0.7599386632, -0.2469136917, 0.2292532078},
            {0.7999516918, -0.1594374882, 0.2831406000},
            {0.8416192397, -0.1258046897, 0.3329573960},
            {0.8885400791,  0.2345363059, 0.3538947488}};
        mat localcorr_q = quantile(algorithm.localCorr(), p, 0);
        REQUIRE(approx_equal(localcorr_q, localcorr_q0, "absdiff", 1e-2));
    }
#endif
}

TEST_CASE("GWCorrelation: cancel")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    mat x = londonhp100_data.cols(2, 3);
    mat y = londonhp100_data.cols(0, 1);
    uword nVar = 4;

    vector<SpatialWeight> spatials;
    vector<GWCorrelation::BandwidthInitilizeType> bandwidthInitialize;
    vector<GWCorrelation::BandwidthSelectionCriterionType> bandwidthSelectionApproach;
    for (size_t i = 0; i < nVar; i++)
    {
        CRSDistance distance;
        BandwidthWeight bandwidth(36, true, BandwidthWeight::Bisquare);
        spatials.push_back(SpatialWeight(&bandwidth, &distance));
        bandwidthInitialize.push_back(GWCorrelation::BandwidthInitilizeType::Null);
        bandwidthSelectionApproach.push_back(GWCorrelation::BandwidthSelectionCriterionType::CV);
    }

    const initializer_list<ParallelType> parallel_list = {
        ParallelType::SerialOnly
#ifdef ENABLE_OPENMP
        ,
        ParallelType::OpenMP
#endif // ENABLE_OPENMP
    };
    auto parallel = GENERATE_REF(values(parallel_list));

    SECTION("GWCorrelation")
    {
        string stage = "GWCorrelation";
        auto progress = GENERATE(0, 10);
        INFO("Settings: " << stage << ", " << progress);

        auto telegram = make_unique<TerminateCheckTelegram>(stage, progress);
        GWCorrelation algorithm;
        algorithm.setTelegram(std::move(telegram));
        algorithm.setCoords(londonhp100_coord);
        algorithm.setVariables2(x);
        algorithm.setVariables1(y);
        algorithm.setSpatialWeights(spatials);
        algorithm.setBandwidthInitilize(bandwidthInitialize);
        algorithm.setBandwidthSelectionApproach(bandwidthSelectionApproach);
        REQUIRE_NOTHROW(algorithm.run());
        REQUIRE(algorithm.status() == Status::Terminated);
    }
}
