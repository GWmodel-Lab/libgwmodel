#include "armadillo_config.h"
#include "mpi.h"

#ifdef ARMA_32BIT_WORD
#define MY_MPI_UWORD MPI_UNSIGNED_LONG
#else   // ARMA_32BIT_WORD
#define MY_MPI_UWORD MPI_UNSIGNED_LONG_LONG
#endif  // ARMA_32BIT_WORD

void mat_mul_mpi(arma::mat& a, arma::mat& b, arma::mat& c, const int ip, const int np, const size_t range);
