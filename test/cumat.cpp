#include <catch2/catch_all.hpp>
#include <armadillo>
#include <iostream>

#include "gwmodelpp/utils/cumat.hpp"

using namespace std;
using namespace arma;

TEST_CASE("cuBLAS gemm")
{
    cublasCreate(&cumat::handle);

    mat A34(3, 4, fill::randu);
    mat A43(4, 3, fill::randu);
    mat A35(3, 5, fill::randu);
    mat B45(4, 5, fill::randu);
    mat B54(5, 4, fill::randu);

    cumat A34cu(A34);
    cumat A43cu(A43);
    cumat A35cu(A35);
    cumat B45cu(B45);
    cumat B54cu(B54);

    SECTION("A * B")
    {
        mat C = A34 * B45;
        cumat U = A34cu * B45cu;
        mat D(size(C));
        U.get(D.memptr());
        REQUIRE(approx_equal(C, D, "absdiff", 1e-6));
    }

    SECTION("A.t * B")
    {
        mat C = A43.t() * B45;
        cumat U = A43cu.t() * B45cu;
        mat D(size(C));
        U.get(D.memptr());
        REQUIRE(approx_equal(C, D, "absdiff", 1e-6));
    }

    SECTION("A * B.t")
    {
        mat C = A35 * B45.t();
        cumat U = A35cu * B45cu.t();
        mat D(size(C));
        U.get(D.memptr());
        REQUIRE(approx_equal(C, D, "absdiff", 1e-6));
    }

    SECTION("A.t * B.t")
    {
        mat C = A43.t() * B54.t();
        cumat U = A43cu.t() * B54cu.t();
        mat D(size(C));
        U.get(D.memptr());
        REQUIRE(approx_equal(C, D, "absdiff", 1e-6));
    }

    cublasDestroy(cumat::handle);
};