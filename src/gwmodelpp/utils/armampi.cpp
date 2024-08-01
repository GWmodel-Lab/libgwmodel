#include "gwmodelpp/utils/armampi.h"
#include <mpi.h>
#include <memory>

using namespace std;
using namespace arma;

void mat_mul_mpi(mat& a, mat& b, mat& c, const int ip, const int np)
{
    arma::uword m = a.n_rows, n = b.n_cols;
    int k = (int) b.n_rows;
    arma::Col<int> b_rows(np, arma::fill::zeros);
    MPI_Allgather(&k, 1, MPI_INT, b_rows.memptr(), 1, MPI_INT, MPI_COMM_WORLD);
    c = mat(m, n, arma::fill::zeros);
    mat a_buf;
    for (int pi = 0; pi < np; pi++)
    {
        arma::Col<int> a_counts = b_rows(pi) * b_rows;
        arma::Col<int> a_disp = arma::cumsum(a_counts) - a_counts;
        a_buf.resize(b_rows(pi), b_rows(ip));
        MPI_Scatterv(a.memptr(), a_counts.mem, a_disp.mem, MPI_DOUBLE, a_buf.memptr(), a_buf.n_elem, MPI_DOUBLE, pi, MPI_COMM_WORLD);
        mat ci = a_buf * b;
        MPI_Reduce(ci.memptr(), c.memptr(), ci.n_elem, MPI_DOUBLE, MPI_SUM, pi, MPI_COMM_WORLD);
    }
}

void mat_quad_mpi(mat& a, mat& aTa, const int ip, const int np)
{
    arma::uword m = a.n_cols, n = a.n_rows;
    int k = (int) a.n_rows;
    arma::Col<int> a_rows(np, arma::fill::zeros);
    MPI_Allgather(&k, 1, MPI_INT, a_rows.memptr(), 1, MPI_INT, MPI_COMM_WORLD);
    aTa = mat(n, m, arma::fill::zeros);
    mat a_buf;
    for (int pi = 0; pi < np; pi++)
    {
        arma::Col<int> a_disp = arma::cumsum(a_rows) - a_rows;
        a_buf = a.cols(a_disp(pi), a_disp(pi) + a_rows(pi) - 1);
        mat aTai = a_buf.t() * a;
        MPI_Reduce(aTai.memptr(), aTa.memptr(), aTai.n_elem, MPI_DOUBLE, MPI_SUM, pi, MPI_COMM_WORLD);
    }
}

double mat_trace_mpi(arma::mat& a, const int ip, const int np)
{
    int k = (int) a.n_rows;
    arma::Col<int> a_rows(np, arma::fill::zeros);
    MPI_Allgather(&k, 1, MPI_INT, a_rows.memptr(), 1, MPI_INT, MPI_COMM_WORLD);
    arma::Col<int> a_disp = arma::cumsum(a_rows) - a_rows;
    mat ai = a.cols(a_disp(ip), a_disp(ip) + a_rows(ip) - 1);
    double ti = trace(ai);
    double tr = 0.0;
    MPI_Allreduce(&ti, &tr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return tr;
}
