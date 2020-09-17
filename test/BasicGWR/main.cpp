#define CATCH_CONFIG_MAIN

#include <catch.hpp>
#include "gwmodel.h"

TEST_CASE("Check create algorithm")
{
    CGwmSpatialAlgorithm* algorithm = gwmodel_create_algorithm();
    REQUIRE(algorithm == nullptr);
}