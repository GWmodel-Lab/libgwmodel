#include "armadillo_config.h"
#include "mpi.h"

#ifdef ARMA_32BIT_WORD
#define GWM_MPI_UWORD MPI_UNSIGNED_LONG
#else   // ARMA_32BIT_WORD
#define GWM_MPI_UWORD MPI_UNSIGNED_LONG_LONG
#endif  // ARMA_32BIT_WORD

void mat_mul_mpi(arma::mat& a, arma::mat& b, arma::mat& c, const int ip, const int np);

void mat_quad_mpi(arma::mat& a, arma::mat& aTa, const int ip, const int np);
