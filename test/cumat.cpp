#include <catch2/catch_all.hpp>
#include <armadillo>
#include <iostream>

#include "gwmodelpp/utils/cumat.hpp"
#include "gwmodelpp/utils/CudaUtils.h"

using namespace std;
using namespace arma;

TEST_CASE("cuBLAS gemm")
{
    cublasCreate(&cumat::handle);

    int m = 3, n = 4, k = 5, s = 5;

    SECTION("mat mul")
    {
        auto flag_A_trans = GENERATE(false, true);
        auto flag_B_trans = GENERATE(false, true);
        INFO("settings: transport A (" << flag_A_trans << ") transport B (" << flag_B_trans << ")");

        mat arA = flag_A_trans ? mat(k, m, arma::fill::randu) : mat(m, k, arma::fill::randu);
        mat arB = flag_B_trans ? mat(n, k, arma::fill::randu) : mat(k, n, arma::fill::randu);
        mat arC = flag_A_trans ? (flag_B_trans ? mat(trans(arA) * trans(arB)) : mat(trans(arA) * arB)) : (flag_B_trans ? mat(arA * trans(arB)) : mat(arA * arB));

        cumat cuA(arA), cuB(arB);
        cumat cuC = flag_A_trans ? (flag_B_trans ? (cuA.t() * cuB.t()) : (cuA.t() * cuB)) : (flag_B_trans ? (cuA * cuB.t()) : (cuA * cuB));

        mat arD(size(arC));
        cuC.get(arD.memptr());
        REQUIRE(approx_equal(arC, arD, "absdiff", 1e-6));
    }

    SECTION("stride mul")
    {
        auto flag_A_trans = GENERATE(false, true);
        auto flag_B_trans = GENERATE(false, true);
        INFO("settings: transport A (" << flag_A_trans << ") transport B (" << flag_B_trans << ")");

        cube arA = flag_A_trans ? cube(k, m, s, arma::fill::randu) : cube(m, k, s, arma::fill::randu);
        cube arB = flag_B_trans ? cube(n, k, s, arma::fill::randu) : cube(k, n, s, arma::fill::randu);
        cube arC(m, n, s);
        for (size_t i = 0; i < s; i++)
        {
            mat a = arA.slice(i), b = arB.slice(i);
            arC.slice(i) = flag_A_trans ? (flag_B_trans ? mat(trans(a) * trans(b)) : mat(trans(a) * b)) : (flag_B_trans ? mat(a * trans(b)) : mat(a * b));
        }
        
        custride cuA(arA), cuB(arB);
        custride cuC = flag_A_trans ? (flag_B_trans ? (cuA.t() * cuB.t()) : (cuA.t() * cuB)) : (flag_B_trans ? (cuA * cuB.t()) : (cuA * cuB));

        cube arD(arC.n_rows, arC.n_cols, arC.n_slices);
        cuC.get(arD.memptr());
        REQUIRE(approx_equal(arC, arD, "absdiff", 1e-6));
    }

    SECTION("stride mat mul")
    {
        auto flag_A_trans = GENERATE(false, true);
        auto flag_B_trans = GENERATE(false, true);
        INFO("settings: transport A (" << flag_A_trans << ") transport B (" << flag_B_trans << ")");

        cube arA = flag_A_trans ? cube(k, m, s, arma::fill::randu) : cube(m, k, s, arma::fill::randu);
        mat arB = flag_B_trans ? mat(n, k, arma::fill::randu) : mat(k, n, arma::fill::randu);;
        cube arC(m, n, s);
        for (size_t i = 0; i < s; i++)
        {
            mat a = arA.slice(i), b = arB;
            arC.slice(i) = flag_A_trans ? (flag_B_trans ? mat(trans(a) * trans(b)) : mat(trans(a) * b)) : (flag_B_trans ? mat(a * trans(b)) : mat(a * b));
        }
        
        custride cuA(arA);
        cumat cuB(arB);
        custride cuC = flag_A_trans ? (flag_B_trans ? (cuA.t() * cuB.t()) : (cuA.t() * cuB)) : (flag_B_trans ? (cuA * cuB.t()) : (cuA * cuB));

        cube arD(arC.n_rows, arC.n_cols, arC.n_slices);
        cuC.get(arD.memptr());
        REQUIRE(approx_equal(arC, arD, "absdiff", 1e-6));
    }

    SECTION("mat stride mul")
    {
        auto flag_A_trans = GENERATE(false, true);
        auto flag_B_trans = GENERATE(false, true);
        INFO("settings: transport A (" << flag_A_trans << ") transport B (" << flag_B_trans << ")");

        mat arA = flag_A_trans ? mat(k, m, arma::fill::randu) : mat(m, k, arma::fill::randu);
        cube arB = flag_B_trans ? cube(n, k, s, arma::fill::randu) : cube(k, n, s, arma::fill::randu);
        cube arC(m, n, s);
        for (size_t i = 0; i < s; i++)
        {
            mat a = arA, b = arB.slice(i);
            arC.slice(i) = flag_A_trans ? (flag_B_trans ? mat(trans(a) * trans(b)) : mat(trans(a) * b)) : (flag_B_trans ? mat(a * trans(b)) : mat(a * b));
        }
        
        cumat cuA(arA);
        custride cuB(arB);
        custride cuC = flag_A_trans ? (flag_B_trans ? (cuA.t() * cuB.t()) : (cuA.t() * cuB)) : (flag_B_trans ? (cuA * cuB.t()) : (cuA * cuB));

        cube arD(arC.n_rows, arC.n_cols, arC.n_slices);
        cuC.get(arD.memptr());
        REQUIRE(approx_equal(arC, arD, "absdiff", 1e-6));
    }

    SECTION("stride inv")
    {
        cube arA(5, 5, 5, arma::fill::randu);
        cube arAI(5, 5, 5);
        for (size_t i = 0; i < 5; i++)
        {
            arAI.slice(i) = inv(arA.slice(i));
        }
        custride sA(arA);
        int* d_info;
        checkCudaErrors(cudaMalloc(&d_info, sizeof(int) * 5));
        custride sAI = sA.inv(d_info);

        cube reAI(5, 5, 5, arma::fill::randu);
        sAI.get(reAI.memptr());
        
        REQUIRE(approx_equal(arAI, reAI, "absdiff", 1e-6));
    }

    cublasDestroy(cumat::handle);
};