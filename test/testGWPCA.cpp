#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <vector>
#include <string>
#include <cstdio>
#include <armadillo>
#include "gwmodelpp/GWPCA.h"
#include "gwmodelpp/spatialweight/CRSDistance.h"
#include "gwmodelpp/spatialweight/BandwidthWeight.h"
#include "gwmodelpp/spatialweight/SpatialWeight.h"
#include "londonhp100.h"
#include "TerminateCheckTelegram.h"

using namespace std;
using namespace arma;
using namespace gwm;

TEST_CASE("GWPCA: basic flow")
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

    mat x = londonhp100_data.cols(1, 3);

    GWPCA algorithm;
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

TEST_CASE("GWPCA: summary statistics")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    CRSDistance distance(false);
    BandwidthWeight bandwidth(15, true, BandwidthWeight::Gaussian);
    SpatialWeight spatial(&bandwidth, &distance);

    mat x = londonhp100_data.cols(1, 3);

    GWPCA algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setVariables(x);
    algorithm.setSpatialWeight(spatial);
    algorithm.setKeepComponents(2);
    REQUIRE_NOTHROW(algorithm.run());

    vec p = {0.0, 0.25, 0.5, 0.75, 1.0};
    
    mat localPV = algorithm.localPV();
    mat sdev = algorithm.sdev();
    mat localVariance = sdev % sdev;
    
    auto bw_weight = algorithm.spatialWeight().weight<BandwidthWeight>();
    double bw_value = bw_weight.bandwidth();
    bool bw_adaptive = bw_weight.adaptive();
    BandwidthWeight::KernelFunctionType kernel_type = bw_weight.kernel();
    const char* bw_kernel_name = "Unknown";
    switch (kernel_type)
    {
        case BandwidthWeight::Gaussian: bw_kernel_name = "Gaussian"; break;
        case BandwidthWeight::Exponential: bw_kernel_name = "Exponential"; break;
        case BandwidthWeight::Bisquare: bw_kernel_name = "Bisquare"; break;
        case BandwidthWeight::Tricube: bw_kernel_name = "Tricube"; break;
        case BandwidthWeight::Boxcar: bw_kernel_name = "Boxcar"; break;
    }
    
    printf("\n");
    printf("Summary of GWPCA information\n");
    printf("=====================================\n\n");
    printf("Bandwidth: %.0f (%s, %s kernel)\n", bw_value, bw_adaptive ? "adaptive" : "fixed", bw_kernel_name);
    printf("\n");
    
    printf("Local variance:\n");
    printf("%-15s %15s %15s\n", "", "Comp.1", "Comp.2");
    mat variance_q = quantile(localVariance, p, 0);
    printf("%-15s %15.3f %15.3f\n", "Min", variance_q(0, 0), variance_q(0, 1));
    printf("%-15s %15.3f %15.3f\n", "1st Qu", variance_q(1, 0), variance_q(1, 1));
    printf("%-15s %15.3f %15.3f\n", "Median", variance_q(2, 0), variance_q(2, 1));
    printf("%-15s %15.3f %15.3f\n", "3rd Qu", variance_q(3, 0), variance_q(3, 1));
    printf("%-15s %15.3f %15.3f\n", "Max", variance_q(4, 0), variance_q(4, 1));
    printf("\n");
    
    printf("Local Proportion of Variance:\n");
    printf("%-15s %15s %15s\n", "", "Comp.1", "Comp.2");
    mat pv_q = quantile(localPV, p, 0);
    printf("%-15s %15.3f %15.3f\n", "Min", pv_q(0, 0), pv_q(0, 1));
    printf("%-15s %15.3f %15.3f\n", "1st Qu", pv_q(1, 0), pv_q(1, 1));
    printf("%-15s %15.3f %15.3f\n", "Median", pv_q(2, 0), pv_q(2, 1));
    printf("%-15s %15.3f %15.3f\n", "3rd Qu", pv_q(3, 0), pv_q(3, 1));
    printf("%-15s %15.3f %15.3f\n", "Max", pv_q(4, 0), pv_q(4, 1));
    
    vec cumulative = sum(localPV, 1);
    vec cum_q = quantile(cumulative, p);
    printf("%-15s %15.3f %15s\n", "Cumulative", cum_q(0), "");
    printf("%-15s %15.3f %15s\n", "", cum_q(1), "");
    printf("%-15s %15.3f %15s\n", "", cum_q(2), "");
    printf("%-15s %15.3f %15s\n", "", cum_q(3), "");
    printf("%-15s %15.3f %15s\n", "", cum_q(4), "");
    printf("\n");
    
    cube scores = algorithm.scores();
    if (scores.n_elem > 0)
    {
        printf("Local Scores:\n");
        printf("%-15s %15s %15s\n", "", "Comp.1", "Comp.2");
        
        mat scores_pc1(scores.n_rows, scores.n_cols);
        mat scores_pc2(scores.n_rows, scores.n_cols);
        for (uword i = 0; i < scores.n_rows; i++)
        {
            for (uword j = 0; j < scores.n_cols; j++)
            {
                scores_pc1(i, j) = scores(i, j, 0);
                if (scores.n_slices > 1)
                {
                    scores_pc2(i, j) = scores(i, j, 1);
                }
            }
        }
        
        vec scores_pc1_flat = vectorise(scores_pc1);
        vec scores_pc2_flat = vectorise(scores_pc2);
        
        vec scores_pc1_q = quantile(scores_pc1_flat, p);
        vec scores_pc2_q = quantile(scores_pc2_flat, p);
        
        printf("%-15s %15.3f %15.3f\n", "Min", scores_pc1_q(0), scores_pc2_q(0));
        printf("%-15s %15.3f %15.3f\n", "1st Qu", scores_pc1_q(1), scores_pc2_q(1));
        printf("%-15s %15.3f %15.3f\n", "Median", scores_pc1_q(2), scores_pc2_q(2));
        printf("%-15s %15.3f %15.3f\n", "3rd Qu", scores_pc1_q(3), scores_pc2_q(3));
        printf("%-15s %15.3f %15.3f\n", "Max", scores_pc1_q(4), scores_pc2_q(4));
        printf("\n");
    }
    else
    {
        printf("Local Scores: (not yet implemented)\n\n");
    }
}


TEST_CASE("GWSS: cancel")
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

    mat x = londonhp100_data.cols(1, 3);

    string stage = "solve";
    auto progress = GENERATE(0, 10);

    auto telegram = make_unique<TerminateCheckTelegram>(stage, progress);
    GWPCA algorithm;
    algorithm.setTelegram(std::move(telegram));
    algorithm.setCoords(londonhp100_coord);
    algorithm.setVariables(x);
    algorithm.setSpatialWeight(spatial);
    algorithm.setKeepComponents(2);
    REQUIRE_NOTHROW(algorithm.run());
    REQUIRE(algorithm.status() == Status::Terminated);
    
}
