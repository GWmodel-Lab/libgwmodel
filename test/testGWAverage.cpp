#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <vector>
#include <string>
#include <armadillo>
#include "gwmodelpp/GWAverage.h"
#include "gwmodelpp/spatialweight/CRSDistance.h"
#include "gwmodelpp/spatialweight/BandwidthWeight.h"
#include "gwmodelpp/spatialweight/SpatialWeight.h"
#include "londonhp100.h"
#include "londonhp.h"
#include "TerminateCheckTelegram.h"

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif // ENABLE_OPENMP

using namespace std;
using namespace arma;
using namespace gwm;

TEST_CASE("GWAverage: londonhp100")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    SECTION("adaptive bandwidth | GWAverage | serial")
    {
        CRSDistance distance(false);
        BandwidthWeight bandwidth(36, true, BandwidthWeight::Gaussian);
        SpatialWeight spatial(&bandwidth, &distance);

        mat x = londonhp100_data.cols(0, 3);

        GWAverage algorithm;
        algorithm.setCoords(londonhp100_coord);
        algorithm.setVariables(x);
        algorithm.setSpatialWeight(spatial);
        REQUIRE_NOTHROW(algorithm.run());

        vec p = {0.0, 0.25, 0.5, 0.75, 1.0};

        mat localmean_q0 = {
            {155530.887621432, 71.3459254279447, 6.92671958853926, 39.0446823327541},
            {163797.287583358, 73.3206754261603, 7.53173813461806, 40.4678577700236},
            {174449.84375947, 74.1174325820277, 7.99672839037902, 43.2051994175928},
            {183893.664229323, 75.3118600659781, 8.59668519607066, 45.2164679493302},
            {188967.723827491, 77.0911277060738, 8.95571485750978, 47.5614366837457}};
        mat localmean_q = quantile(algorithm.localMean(), p, 0);
        REQUIRE(approx_equal(localmean_q, localmean_q0, "absdiff", 1e-8));

        mat localsdev_q0 = {
            {72593.9921817404, 28.3099770356131, 2.01116607468286, 8.4638277922896},
            {77170.1777015588, 29.7600378924393, 2.30743344312421, 9.96621298558318},
            {80165.9773579845, 30.1391577420805, 2.38179570120204, 10.7335766559347},
            {83051.0304538234, 31.1706493250992, 2.55775279583101, 11.2114283718788},
            {86969.3221725472, 32.2881606484993, 2.73655762029611, 12.0280808931404}};
        mat localsdev_q = quantile(algorithm.localSDev(), p, 0);
        REQUIRE(approx_equal(localsdev_q, localsdev_q0, "absdiff", 1e-8));

        mat localskew_q0 = {
            {1.4092924720023, 1.37071524667066, -0.555779877884807, -0.111701647027156},
            {1.52855813553472, 1.41818465286691, -0.306523936775326, 0.0688874355072382},
            {1.7202801518175, 1.48329930159006, 0.00225908149445752, 0.346270682414352},
            {2.01494028420899, 1.65657212304686, 0.278551178439127, 0.529519631010928},
            {2.29647902712578, 1.85491109594693, 0.456786660462276, 0.614689755724046}};
        mat localskew_q = quantile(algorithm.localSkewness(), p, 0);
        REQUIRE(approx_equal(localskew_q, localskew_q0, "absdiff", 1e-8));

        mat localcv_q0 = {
            {0.426450737915543, 0.393305654809352, 0.224567899568219, 0.211056223726344},
            {0.437470421902242, 0.405332689971102, 0.272318025735197, 0.230225532145268},
            {0.451648453541485, 0.407743814376727, 0.314108382503925, 0.23750624244725},
            {0.490606242780692, 0.411684812307897, 0.334007325644591, 0.26722230789406},
            {0.520406719170266, 0.419294146622537, 0.339426249173607, 0.2901354343712}};
        mat localcv_q = quantile(algorithm.localCV(), p, 0);
        REQUIRE(approx_equal(localcv_q, localcv_q0, "absdiff", 1e-8));
    }

    SECTION("adaptive bandwidth | GWAverage | calibration | serial")
    {

        CRSDistance distance(false);
        BandwidthWeight bandwidth(36, true, BandwidthWeight::Gaussian);
        SpatialWeight spatial(&bandwidth, &distance);

        mat x = londonhp100_data.cols(0, 3);
        mat locations = londonhp100_coord.rows(0,49);

        GWAverage algorithm;
        algorithm.setCoords(londonhp100_coord);
        algorithm.setSpatialWeight(spatial);
        REQUIRE_NOTHROW(algorithm.calibration(locations, x));

        // vec p = {0.0, 0.5, 1.0};

        // mat localmean_q0 = {
        //     {163719.862916961, 76.7798124032325, 7.21724926005143, 42.9934294704426},
        //     {171656.123264602, 77.4498277745430, 7.83847211370497, 44.4684218168923},
        //     {178159.523929681, 78.1330047900227, 8.19568286692306, 46.6818015195459}};
        // mat localmean_q = quantile(algorithm.localMean(), p, 0);
        // localmean_q.print();
        // REQUIRE(approx_equal(localmean_q, localmean_q0, "absdiff", 1e-6));

        // mat localsdev_q0 = {
        //     {76138.0546932983, 29.7372887770954, 2.21844086330195, 8.81825019566705},
        //     {77116.4575192220, 30.3549665798213, 2.34940000365892, 9.55234528534620},
        //     {78329.5185515576, 31.0371475770194, 2.48272333683410, 10.33289050254658}};
        // mat localsdev_q = quantile(algorithm.localSDev(), p, 0);
        // REQUIRE(approx_equal(localsdev_q, localsdev_q0, "absdiff", 1e-6));

        // mat localcv_q0 = {
        //     {0.437242859717688, 0.383755183129257, 0.271034835842149, 0.204563926621541},
        //     {0.449541358593578, 0.393089483221880, 0.299811193591396, 0.214471653323718},
        //     {0.467977489025189, 0.400985640582967, 0.343931475958593, 0.221623023843879}};
        // mat localcv_q = quantile(algorithm.localCV(), p, 0);
        // REQUIRE(approx_equal(localcv_q, localcv_q0, "absdiff", 1e-6));
    }

#ifdef ENABLE_OPENMP
    SECTION("adaptive bandwidth | GWAverage | omp parallel")
    {
        CRSDistance distance(false);
        BandwidthWeight bandwidth(36, true, BandwidthWeight::Gaussian);
        SpatialWeight spatial(&bandwidth, &distance);

        mat x = londonhp100_data.cols(0, 3);

        GWAverage algorithm;
        algorithm.setCoords(londonhp100_coord);
        algorithm.setVariables(x);
        algorithm.setSpatialWeight(spatial);
        algorithm.setParallelType(ParallelType::OpenMP);
        algorithm.setOmpThreadNum(omp_get_num_threads());
        REQUIRE_NOTHROW(algorithm.run());

        vec p = {0.0, 0.25, 0.5, 0.75, 1.0};

        mat localmean_q0 = {
            {155530.887621432, 71.3459254279447, 6.92671958853926, 39.0446823327541},
            {163797.287583358, 73.3206754261603, 7.53173813461806, 40.4678577700236},
            {174449.84375947, 74.1174325820277, 7.99672839037902, 43.2051994175928},
            {183893.664229323, 75.3118600659781, 8.59668519607066, 45.2164679493302},
            {188967.723827491, 77.0911277060738, 8.95571485750978, 47.5614366837457}};
        mat localmean_q = quantile(algorithm.localMean(), p, 0);
        REQUIRE(approx_equal(localmean_q, localmean_q0, "absdiff", 1e-8));

        mat localsdev_q0 = {
            {72593.9921817404, 28.3099770356131, 2.01116607468286, 8.4638277922896},
            {77170.1777015588, 29.7600378924393, 2.30743344312421, 9.96621298558318},
            {80165.9773579845, 30.1391577420805, 2.38179570120204, 10.7335766559347},
            {83051.0304538234, 31.1706493250992, 2.55775279583101, 11.2114283718788},
            {86969.3221725472, 32.2881606484993, 2.73655762029611, 12.0280808931404}};
        mat localsdev_q = quantile(algorithm.localSDev(), p, 0);
        REQUIRE(approx_equal(localsdev_q, localsdev_q0, "absdiff", 1e-8));

        mat localskew_q0 = {
            {1.4092924720023, 1.37071524667066, -0.555779877884807, -0.111701647027156},
            {1.52855813553472, 1.41818465286691, -0.306523936775326, 0.0688874355072382},
            {1.7202801518175, 1.48329930159006, 0.00225908149445752, 0.346270682414352},
            {2.01494028420899, 1.65657212304686, 0.278551178439127, 0.529519631010928},
            {2.29647902712578, 1.85491109594693, 0.456786660462276, 0.614689755724046}};
        mat localskew_q = quantile(algorithm.localSkewness(), p, 0);
        REQUIRE(approx_equal(localskew_q, localskew_q0, "absdiff", 1e-8));

        mat localcv_q0 = {
            {0.426450737915543, 0.393305654809352, 0.224567899568219, 0.211056223726344},
            {0.437470421902242, 0.405332689971102, 0.272318025735197, 0.230225532145268},
            {0.451648453541485, 0.407743814376727, 0.314108382503925, 0.23750624244725},
            {0.490606242780692, 0.411684812307897, 0.334007325644591, 0.26722230789406},
            {0.520406719170266, 0.419294146622537, 0.339426249173607, 0.2901354343712}};
        mat localcv_q = quantile(algorithm.localCV(), p, 0);
        REQUIRE(approx_equal(localcv_q, localcv_q0, "absdiff", 1e-8));
    }
#endif

}

TEST_CASE("GWAverage: cancel")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    CRSDistance distance(false);
    BandwidthWeight bandwidth(36, true, BandwidthWeight::Gaussian);
    SpatialWeight spatial(&bandwidth, &distance);

    mat x = londonhp100_data.cols(0, 3);

    const initializer_list<ParallelType> parallel_list = {
        ParallelType::SerialOnly
#ifdef ENABLE_OPENMP
        , ParallelType::OpenMP
#endif // ENABLE_OPENMP     
    };
    auto parallel = GENERATE_REF(values(parallel_list));

    SECTION("average")
    {
        string stage = "GWAverage";
        auto progress = GENERATE(0, 10);
        INFO("Settings: " << stage << ", " << progress);

        auto telegram = make_unique<TerminateCheckTelegram>(stage, progress);
        GWAverage algorithm;
        algorithm.setTelegram(std::move(telegram));
        algorithm.setCoords(londonhp100_coord);
        algorithm.setVariables(x);
        algorithm.setSpatialWeight(spatial);
        REQUIRE_NOTHROW(algorithm.run());
        REQUIRE(algorithm.status() == Status::Terminated);
    }
    
}