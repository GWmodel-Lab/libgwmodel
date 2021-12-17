#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <vector>
#include <string>
#include <armadillo>
#include <gwmodel.h>
#include "londonhp100.h"

using namespace std;
using namespace arma;

mat interface2mat(const GwmMatInterface& interface)
{
    return mat(interface.data, interface.rows, interface.cols);
}

TEST_CASE("GWSS: basic flow")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    GwmMatInterface londonhp100_coord_interface; 
    londonhp100_coord_interface.rows = londonhp100_coord.n_rows;
    londonhp100_coord_interface.cols = londonhp100_coord.n_cols;
    londonhp100_coord_interface.data = londonhp100_coord.mem;
    GwmMatInterface londonhp100_data_interface;
    londonhp100_data_interface.rows = londonhp100_data.n_rows;
    londonhp100_data_interface.cols = londonhp100_data.n_cols;
    londonhp100_data_interface.data = londonhp100_data.mem;
    GwmNameListInterface londonhp100_field_interface;
    londonhp100_field_interface.size = londonhp100_fields.size();
    londonhp100_field_interface.items = new GwmNameInterface[londonhp100_field_interface.size];
    for (int i = 0; i < londonhp100_fields.size(); i++)
    {
        strcpy(londonhp100_field_interface.items[i], londonhp100_fields[i].data());
    }
    CGwmSimpleLayer* londonhp = gwmodel_create_simple_layer(londonhp100_coord_interface, londonhp100_data_interface, londonhp100_field_interface);

    CGwmDistance* distance = gwmodel_create_crs_distance(false);
    CGwmWeight* bandwidth = gwmodel_create_bandwidth_weight(36, true, KernelFunctionType::Gaussian);
    CGwmSpatialWeight* spatial = gwmodel_create_spatial_weight(distance, bandwidth);

    GwmVariableInterface floorsz;
    floorsz.index = 1;
    floorsz.isNumeric = true;
    strcpy(floorsz.name, "FLOORSZ");
    GwmVariableInterface unemploy;
    unemploy.index = 2;
    unemploy.isNumeric = true;
    strcpy(unemploy.name, "UNEMPLOY");
    GwmVariableInterface prof;
    prof.index = 3;
    prof.isNumeric = true;
    strcpy(prof.name, "PROF");
    GwmVariableListInterface vars;
    vars.size = 3;
    vars.items = new GwmVariableInterface[vars.size];
    vars.items[0] = floorsz;
    vars.items[1] = unemploy;
    vars.items[2] = prof;

    CGwmGWPCA* algorithm = gwmodel_create_gwpca_algorithm();
    gwmodel_set_gwpca_source_layer(algorithm, londonhp);
    gwmodel_set_gwpca_variables(algorithm, vars);
    gwmodel_set_gwpca_spatial_weight(algorithm, spatial);
    gwmodel_set_gwpca_options(algorithm, 2);
    REQUIRE_NOTHROW(gwmodel_run_gwpca(algorithm));

    vec p = {0.0, 0.25, 0.5, 0.75, 1.0};

    mat comp_q0 = {
        { 86.09381920388,7.38948790899526 },
        { 87.2417310474256,10.0805823313445 },
        { 88.5114946422145,11.4166428700704 },
        { 89.8514496001622,12.6890545321313 },
        { 92.5449003124064,13.8382823156345 }
    };
    GwmMatInterface comp = gwmodel_get_gwpca_local_pv(algorithm);
    mat comp_q = quantile(interface2mat(comp), p, 0);
    REQUIRE(approx_equal(comp_q, comp_q0, "absdiff", 1e-8));

    mat loadings_pc1_q0 = {
        { 0.997738665168933,-0.0115292388648388,-0.040450830035712 },
        { 0.998673840689833,-0.00822122467004101,-0.00468318323510321 },
        { 0.999297415085303,-0.00389424492785976,0.0320948265474252 },
        { 0.999678999646886,0.00274831974093292,0.0508510246498129 },
        { 0.999999194544384,0.0105326992413149,0.0662213367046487 }
    };
    GwmMatInterface loadings_pc1 = gwmodel_get_gwpca_loadings(algorithm, 0);
    mat loadings_pc1_q = quantile(interface2mat(loadings_pc1), p, 0);
    REQUIRE(approx_equal(loadings_pc1_q, loadings_pc1_q0, "absdiff", 1e-8));

    mat loadings_pc2_q0 = {
        { -0.0671714511032381,-0.219162658504117,-0.97924877722135 },
        { -0.0513759501838017,-0.214853304247932,0.976144875457391 },
        { -0.0321827960857794,-0.211329933955831,0.976665314794129 },
        {  0.00517581158157478,-0.204353440937033,0.978464099165948 },
        {  0.0417635544237787,0.202661857194208,0.980384688418526 }
    };
    GwmMatInterface loadings_pc2 = gwmodel_get_gwpca_loadings(algorithm, 1);
    mat loadings_pc2_q = quantile(interface2mat(loadings_pc2), p, 0);
    REQUIRE(approx_equal(loadings_pc2_q, loadings_pc2_q0, "absdiff", 1e-8));

    REQUIRE_NOTHROW(gwmodel_delete_mat(&comp));
    REQUIRE_NOTHROW(gwmodel_delete_mat(&loadings_pc1));
    REQUIRE_NOTHROW(gwmodel_delete_mat(&loadings_pc2));
    REQUIRE_NOTHROW(gwmodel_delete_gwpca_algorithm(algorithm));
    REQUIRE_NOTHROW(gwmodel_delete_string_list(&londonhp100_field_interface));
    REQUIRE_NOTHROW(gwmodel_delete_variable_list(&vars));
}
