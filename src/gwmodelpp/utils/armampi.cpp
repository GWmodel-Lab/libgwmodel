#include "gwmodelpp/utils/armampi.h"
#include <mpi.h>
#include <memory>

using namespace std;
using namespace arma;

void mat_mul_mpi(mat& a, mat& b, mat& c, const int ip, const int np, const size_t range)
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
