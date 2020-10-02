#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <vector>
#include <string>
#include <armadillo>
#include <gwmodel.h>
#include "londonhp100.h"

using namespace std;
using namespace arma;

vector<int> variables2indices(const GwmVariableListInterface& variables)
{
    vector<int> index;
    for (size_t i = 0; i < variables.size; i++)
    {
        index.push_back(variables.items[i].index);
    }
    return index;
}

TEST_CASE("BasicGWR: Multithread")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    GwmMatInterface londonhp100_coord_interface = { londonhp100_coord.n_rows, londonhp100_coord.n_cols, londonhp100_coord.memptr() };
    GwmMatInterface londonhp100_data_interface = { londonhp100_data.n_rows, londonhp100_data.n_cols, londonhp100_data.memptr() };
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

    GwmVariableInterface purchase = { 0, true, "PURCHASE" };
    GwmVariableInterface floorsz = { 1, true, "FLOORSZ" };
    GwmVariableInterface unemploy = { 2, true, "UNEMPLOY" };
    GwmVariableInterface prof = { 3, true, "PROF" };
    GwmVariableListInterface indepVars;
    indepVars.size = 3;
    indepVars.items = new GwmVariableInterface[indepVars.size];
    indepVars.items[0] = floorsz;
    indepVars.items[1] = unemploy;
    indepVars.items[2] = prof;

    CGwmGWRBasic* algorithm = gwmodel_create_gwr_algorithm();
    gwmodel_set_gwr_source_layer(algorithm, londonhp);
    gwmodel_set_gwr_dependent_variable(algorithm, purchase);
    gwmodel_set_gwr_independent_variable(algorithm, indepVars);
    gwmodel_set_gwr_spatial_weight(algorithm, spatial);
    gwmodel_set_gwr_options(algorithm, true);
    gwmodel_set_gwr_indep_vars_autoselection(algorithm, 3.0);
    gwmodel_set_gwr_bandwidth_autoselection(algorithm, BandwidthSelectionCriterionType::CV);
    gwmodel_set_gwr_openmp(algorithm, 8);
    REQUIRE_NOTHROW(gwmodel_run_gwr(algorithm));

    GwmVariablesCriterionListInterface criterionList = gwmodel_get_gwr_indep_var_criterions(algorithm);
    
    REQUIRE_THAT(variables2indices(criterionList.items[0].variables), Catch::Equals(vector<int>({ 2 })));
    REQUIRE(strcmp(criterionList.items[0].variables.items[0].name, "UNEMPLOY") == 0);
    REQUIRE_THAT(criterionList.items[0].criterion, Catch::WithinAbs(2551.61359020599, 1e-8));
    
    REQUIRE_THAT(variables2indices(criterionList.items[1].variables), Catch::Equals(vector<int>({ 3 })));
    REQUIRE_THAT(criterionList.items[1].criterion, Catch::WithinAbs(2551.30032201349, 1e-8));
    REQUIRE(strcmp(criterionList.items[1].variables.items[0].name, "PROF") == 0);

    REQUIRE_THAT(variables2indices(criterionList.items[2].variables), Catch::Equals(vector<int>({ 1 })));
    REQUIRE(strcmp(criterionList.items[2].variables.items[0].name, "FLOORSZ") == 0);
    REQUIRE_THAT(criterionList.items[2].criterion, Catch::WithinAbs(2468.93236280013, 1e-8));

    REQUIRE_THAT(variables2indices(criterionList.items[3].variables), Catch::Equals(vector<int>({ 1, 3 })));
    REQUIRE(strcmp(criterionList.items[3].variables.items[0].name, "FLOORSZ") == 0);
    REQUIRE(strcmp(criterionList.items[3].variables.items[1].name, "PROF") == 0);
    REQUIRE_THAT(criterionList.items[3].criterion, Catch::WithinAbs(2452.86447942033, 1e-8));

    REQUIRE_THAT(variables2indices(criterionList.items[4].variables), Catch::Equals(vector<int>({ 1, 2 })));
    REQUIRE(strcmp(criterionList.items[4].variables.items[0].name, "FLOORSZ") == 0);
    REQUIRE(strcmp(criterionList.items[4].variables.items[1].name, "UNEMPLOY") == 0);
    REQUIRE_THAT(criterionList.items[4].criterion, Catch::WithinAbs(2450.59642666509, 1e-8));

    REQUIRE_THAT(variables2indices(criterionList.items[5].variables), Catch::Equals(vector<int>({ 1, 2, 3 })));
    REQUIRE(strcmp(criterionList.items[5].variables.items[0].name, "FLOORSZ") == 0);
    REQUIRE(strcmp(criterionList.items[5].variables.items[1].name, "UNEMPLOY") == 0);
    REQUIRE(strcmp(criterionList.items[5].variables.items[2].name, "PROF") == 0);
    REQUIRE_THAT(criterionList.items[5].criterion, Catch::WithinAbs(2452.80388934625, 1e-8));

    CGwmSpatialWeight* spatial_new = gwmodel_get_gwr_spatial_weight(algorithm);

    GwmBandwidthKernelInterface bw_new_interf;
    if (gwmodel_get_spatial_bandwidth_weight(spatial_new, &bw_new_interf))
    {
        REQUIRE(bw_new_interf.size == 31);
    }
    else
    {
        FAIL("Cannot regard a CGwmWeight instance as a CGwmBandwidthWeight instance!");
    }

    GwmRegressionDiagnostic diagnostic = gwmodel_get_gwr_diagnostic(algorithm);
    REQUIRE_THAT(diagnostic.AIC, Catch::WithinAbs(2435.8161441795, 1e-8));
    REQUIRE_THAT(diagnostic.AICc, Catch::WithinAbs(2445.49629974057, 1e-8));
    REQUIRE_THAT(diagnostic.RSquare, Catch::WithinAbs(0.706143867720706, 1e-8));
    REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::WithinAbs(0.678982114793865, 1e-8));

    REQUIRE_NOTHROW(gwmodel_delete_gwr_algorithm(algorithm));
    REQUIRE_NOTHROW(gwmodel_delete_string_list(&londonhp100_field_interface));
    REQUIRE_NOTHROW(gwmodel_delete_variable_list(&indepVars));
}
