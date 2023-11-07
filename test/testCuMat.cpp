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

        cumat cuA(arA), cuB(arB), cuC(arC.n_rows, arC.n_cols);
        if (flag_A_trans) 
        {
            if (flag_B_trans) 
            {
                cuC = cuA.t() * cuB.t();
            }
            else
            {
                cuC = cuA.t() * cuB;
            }
        }
        else 
        {
            if (flag_B_trans)
            {
                cuC = cuA * cuB.t();
            }
            else
            {
                cuC = cuA * cuB;
            }
        }

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
        
        custride cuA(arA), cuB(arB), cuC(arC.n_rows, arC.n_cols, arC.n_slices);
        if (flag_A_trans) 
        {
            if (flag_B_trans) 
            {
                cuC = cuA.t() * cuB.t();
            }
            else
            {
                cuC = cuA.t() * cuB;
            }
        }
        else 
        {
            if (flag_B_trans)
            {
                cuC = cuA * cuB.t();
            }
            else
            {
                cuC = cuA * cuB;
            }
        }

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
        custride cuC(arC.n_rows, arC.n_cols, arC.n_slices);
        if (flag_A_trans) 
        {
            if (flag_B_trans) 
            {
                cuC = cuA.t() * cuB.t();
            }
            else
            {
                cuC = cuA.t() * cuB;
            }
        }
        else 
        {
            if (flag_B_trans)
            {
                cuC = cuA * cuB.t();
            }
            else
            {
                cuC = cuA * cuB;
            }
        }

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
        custride cuC(arC.n_rows, arC.n_cols, arC.n_slices);
        if (flag_A_trans) 
        {
            if (flag_B_trans) 
            {
                cuC = cuA.t() * cuB.t();
            }
            else
            {
                cuC = cuA.t() * cuB;
            }
        }
        else 
        {
            if (flag_B_trans)
            {
                cuC = cuA * cuB.t();
            }
            else
            {
                cuC = cuA * cuB;
            }
        }

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

TEST_CASE("Mat mul benchmark")
{
    size_t n = 10000, k = 4, g = 100;
    mat ar_x(n, k, arma::fill::randn);
    vec ar_w(n, arma::fill::randu);
    size_t groups = n / 100;

    BENCHMARK("simulate | arma")
    {
        cube xtwx_inv(k, k, n);
        for (size_t i = 0; i < n; i++)
        {
            xtwx_inv.slice(i) = inv((ar_x.each_col() % ar_w).t() * ar_x);
        }
        return xtwx_inv;
    };
    
    BENCHMARK("simulate | cublas")
    {
        cublasCreate(&cumat::handle);
        int *d_info, *p_info;
        p_info = new int[g];
        cudaMalloc(&d_info, sizeof(int) * g);
        double *d_x, *d_w;
        cudaMalloc(&d_x, n * k * sizeof(double));
        cudaMalloc(&d_w, n * sizeof(double));
        cudaMemcpy(d_x, ar_x.mem, n * k * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_w, ar_w.mem, n * sizeof(double), cudaMemcpyHostToDevice);
        double *ds_xtwx, *ds_xtwx_inv;
        cudaMalloc(&ds_xtwx, k * k * g * sizeof(double));
        cudaMalloc(&ds_xtwx_inv, k * k * n * sizeof(double));
        double **pd_xtwx = new double*[g], **pd_xtwx_inv = new double*[n];
        for (size_t i = 0; i < g; i++)
        {
            pd_xtwx[i] = ds_xtwx + k * k * i;
        }
        for (size_t i = 0; i < n; i++)
        {
            pd_xtwx_inv[i] = ds_xtwx_inv + k * k * i;
        }
        double **dd_xtwx, **dd_xtwx_inv;
        cudaMalloc(&dd_xtwx, g * sizeof(double*));
        cudaMalloc(&dd_xtwx_inv, n * sizeof(double*));
        cudaMemcpy(dd_xtwx, pd_xtwx, g * sizeof(double*), cudaMemcpyHostToDevice);
        cudaMemcpy(dd_xtwx_inv, pd_xtwx_inv, n * sizeof(double*), cudaMemcpyHostToDevice);
        double *ds_xtw;
        cudaMalloc(&ds_xtw, n * k * g * sizeof(double));
        for (size_t j = 0; j < groups; j++)
        {
            for (size_t i = 0; i < g; i++)
            {
                cublasDdgmm(cubase::handle, CUBLAS_SIDE_RIGHT, n, k, d_x, n, d_w, 1, ds_xtw + n * k * i, n);
            }
            cublasDgemmStridedBatched(
                cubase::handle, CUBLAS_OP_T, CUBLAS_OP_N,
                k, k, n, &cubase::alpha1,
                ds_xtw, n, n * k,
                d_x, n, 0,
                &cubase::beta0, ds_xtwx, k, k * k, g
            );
            cublasDmatinvBatched(cubase::handle, k, dd_xtwx, k, dd_xtwx_inv + j * g, k, d_info, g);
        }
        cudaFree(d_info);
        cudaFree(d_x);
        cudaFree(d_w);
        cudaFree(ds_xtwx);
        cudaFree(ds_xtwx_inv);
        cudaFree(dd_xtwx);
        cudaFree(dd_xtwx_inv);
        cudaFree(ds_xtw);
        delete[] pd_xtwx_inv;
        delete[] pd_xtwx;
        delete[] p_info;
        cublasDestroy(cumat::handle);
        return dd_xtwx_inv;
    };
    
    BENCHMARK("simulate | cumat")
    {
        cublasCreate(&cumat::handle);
        cumat cu_x(ar_x);
        cumat cu_w(ar_w);
        int *d_info, *p_info;
        p_info = new int[g];
        cudaMalloc(&d_info, sizeof(int) * g);
        custride xtwx(k, k, g), xtwx_inv(k, k, n);
        cumat xtw(cu_x.nrows(), cu_x.ncols());
        for (size_t j = 0; j < groups; j++)
        {
            for (size_t i = 0; i < g; i++)
            {
                xtw = cu_x.diagmul(cu_w);
                xtwx.strides(i) = xtw.t() * cu_x;
            }
            xtwx_inv.strides(j * g, j * g + g) = xtwx.inv(d_info);
        }
        cudaFree(d_info);
        delete[] p_info;
        cublasDestroy(cumat::handle);
        return xtwx_inv;
    };
};