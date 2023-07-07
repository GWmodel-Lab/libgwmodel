#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <vector>
#include <string>
#include <armadillo>
#include "gwmodelpp/GWDA.h"
#include "gwmodelpp/spatialweight/CRSDistance.h"
#include "gwmodelpp/spatialweight/BandwidthWeight.h"
#include "gwmodelpp/spatialweight/SpatialWeight.h"
#include "londonhp.h"

using namespace std;
using namespace arma;
using namespace gwm;

template<typename T>
vector<string> convertY(vector<T> &y1){
    uword n=y1.size();
    vector<string> y(n);
    for(uword i=0;i<n;i++){
        y[i]=to_string(y1[i]);
    }
    return y;
}

//template<class T>
TEST_CASE("GWDA: LondonHP")
{
    mat londonhp_coord, londonhp_data;
    vector<string> londonhp_fields;
    if (!read_londonhp(londonhp_coord, londonhp_data, londonhp_fields))
    {
        FAIL("Cannot load londonhp data.");
    }

    CRSDistance distance(false);

    mat x1 = londonhp_data.cols(17, 18);
    mat x2=londonhp_data.col(1);
    mat x=join_rows(x2,x1);
    vec y0=londonhp_data.col(13);
    vector<uword> y1;
    y1=conv_to<vector<uword>>::from(y0);
    vector<string> y;
    y=convertY(y1);
    //vector<string> y=conv_to<std::vector<string>>::from(y1);
    

    GWDA algorithm;
    algorithm.setCoords(londonhp_coord);
    algorithm.setVariables(x);
    algorithm.setGroup(y);

    SECTION("adaptive bandwidth | no bandwidth optimization | no variable optimization | serial") {
        BandwidthWeight bandwidth(36, true, BandwidthWeight::Gaussian);
        SpatialWeight spatial(&bandwidth, &distance);
        algorithm.setSpatialWeight(spatial);
        //algorithm.setIsWqda(true);
        REQUIRE_NOTHROW(algorithm.run());
        REQUIRE_THAT(algorithm.correctRate(), Catch::Matchers::WithinAbs(0.905063291139, 1e-8));
        //REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2448.27206524754, 1e-8));
        //REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.708010632044736, 1e-8));
        //REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.674975341723766, 1e-8));
    }
    SECTION("adaptive bandwidth | no bandwidth optimization | no variable optimization | serial || wqda") {
        BandwidthWeight bandwidth(36, true, BandwidthWeight::Gaussian);
        SpatialWeight spatial(&bandwidth, &distance);
        algorithm.setSpatialWeight(spatial);
        algorithm.setIsWqda(true);
        REQUIRE_NOTHROW(algorithm.run());
        REQUIRE_THAT(algorithm.correctRate(), Catch::Matchers::WithinAbs(0.89873417721519, 1e-8));
        //REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2448.27206524754, 1e-8));
        //REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.708010632044736, 1e-8));
        //REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.674975341723766, 1e-8));
    }
    
}
/*
TEST_CASE("GWSS: correlation with first variable only")
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

    vec y = londonhp100_data.col(0);
    mat x = londonhp100_data.cols(0, 3);

    GWSS algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setVariables(x);
    algorithm.setSpatialWeight(spatial);
    algorithm.setIsCorrWithFirstOnly(true);
    REQUIRE_NOTHROW(algorithm.run());

    vec p = {0.0, 0.25, 0.5, 0.75, 1.0};

    mat localcorr_q0 = {
        {0.748948486801849,-0.320600183598632,0.203011140141453},
        {0.762547101624896,-0.297490396583388,0.246560726908457},
        {0.78483823103956,-0.254451851221453,0.282830629241902},
        {0.809708575169509,-0.240880285636795,0.324300091997221},
        {0.838005736892351,-0.201907496636598,0.35265748682446}
    };
    mat localcorr_q = quantile(algorithm.localCorr(), p, 0);
    REQUIRE(approx_equal(localcorr_q, localcorr_q0, "absdiff", 1e-8));
}

#ifdef ENABLE_OPENMP
TEST_CASE("GWSS: multithread basic flow")
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

    vec y = londonhp100_data.col(0);
    mat x = londonhp100_data.cols(0, 3);

    GWSS algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setVariables(x);
    algorithm.setSpatialWeight(spatial);
    algorithm.setParallelType(ParallelType::OpenMP);
    algorithm.setOmpThreadNum(6);
    REQUIRE_NOTHROW(algorithm.run());

    vec p = {0.0, 0.25, 0.5, 0.75, 1.0};

    mat localmean_q0 = {
        {155530.887621432,71.3459254279447,6.92671958853926,39.0446823327541},
        {163797.287583358,73.3206754261603,7.53173813461806,40.4678577700236},
        {174449.84375947,74.1174325820277,7.99672839037902,43.2051994175928},
        {183893.664229323,75.3118600659781,8.59668519607066,45.2164679493302},
        {188967.723827491,77.0911277060738,8.95571485750978,47.5614366837457}
    };
    mat localmean_q = quantile(algorithm.localMean(), p, 0);
    REQUIRE(approx_equal(localmean_q, localmean_q0, "absdiff", 1e-8));

    mat localsdev_q0 = {
        {72593.9921817404,28.3099770356131,2.01116607468286,8.4638277922896},
        {77170.1777015588,29.7600378924393,2.30743344312421,9.96621298558318},
        {80165.9773579845,30.1391577420805,2.38179570120204,10.7335766559347},
        {83051.0304538234,31.1706493250992,2.55775279583101,11.2114283718788},
        {86969.3221725472,32.2881606484993,2.73655762029611,12.0280808931404}
    };
    mat localsdev_q = quantile(algorithm.localSDev(), p, 0);
    REQUIRE(approx_equal(localsdev_q, localsdev_q0, "absdiff", 1e-8));

    mat localskew_q0 = {
        {1.4092924720023,1.37071524667066,-0.555779877884807,-0.111701647027156},
        {1.52855813553472,1.41818465286691,-0.306523936775326,0.0688874355072382},
        {1.7202801518175,1.48329930159006,0.00225908149445752,0.346270682414352},
        {2.01494028420899,1.65657212304686,0.278551178439127,0.529519631010928},
        {2.29647902712578,1.85491109594693,0.456786660462276,0.614689755724046}
    };
    mat localskew_q = quantile(algorithm.localSkewness(), p, 0);
    REQUIRE(approx_equal(localskew_q, localskew_q0, "absdiff", 1e-8));

    mat localcv_q0 = {
        {0.426450737915543,0.393305654809352,0.224567899568219,0.211056223726344},
        {0.437470421902242,0.405332689971102,0.272318025735197,0.230225532145268},
        {0.451648453541485,0.407743814376727,0.314108382503925,0.23750624244725},
        {0.490606242780692,0.411684812307897,0.334007325644591,0.26722230789406},
        {0.520406719170266,0.419294146622537,0.339426249173607,0.2901354343712}
    };
    mat localcv_q = quantile(algorithm.localCV(), p, 0);
    REQUIRE(approx_equal(localcv_q, localcv_q0, "absdiff", 1e-8));

    mat localcorr_q0 = {
        {0.748948486801849,-0.320600183598632,0.203011140141453,-0.126882976445561,-0.0892568204410789,-0.948799446008617},
        {0.762547101624896,-0.297490396583388,0.246560726908457,-0.0855566960390598,-0.0108045673038358,-0.939669443787772},
        {0.78483823103956,-0.254451851221453,0.282830629241902,-0.0466716717586483,0.0860667834905564,-0.930939912151655},
        {0.809708575169509,-0.240880285636795,0.324300091997221,0.030563100930792,0.14224438342357,-0.924880287887928},
        {0.838005736892351,-0.201907496636598,0.35265748682446,0.106759558870671,0.161751404622356,-0.906594939821811}
    };
    mat localcorr_q = quantile(algorithm.localCorr(), p, 0);
    REQUIRE(approx_equal(localcorr_q, localcorr_q0, "absdiff", 1e-8));

    mat localscorr_q0 = {
        {0.521222457142438,-0.386537315399977,0.272098100316185,-0.132913057346789,-0.0706904961467669,-0.940629495956178},
        {0.546058956484106,-0.367722715928213,0.28224553968716,-0.100607936868221,0.00362924865128611,-0.931506008150178},
        {0.591076906824072,-0.333869710084257,0.336014460751443,-0.0756778419096376,0.0739387878352967,-0.928259365660612},
        {0.642395389246104,-0.314342558536871,0.358117991041394,-0.000170454849373912,0.108476590000141,-0.915878602985333},
        {0.685066744419873,-0.296544286394518,0.380785226148097,0.0690739762835091,0.170298974146835,-0.895252623185884}
    };
    mat localscorr_q = quantile(algorithm.localSCorr(), p, 0);
    REQUIRE(approx_equal(localscorr_q, localscorr_q0, "absdiff", 1e-1));
}
#endif*/
