#ifndef ARMADILLO_CONFIG
#define ARMADILLO_CONFIG

#ifdef USE_RCPPARMADILLO
#include <RcppArmadillo.h>
#else   // USE_RCPPARMADILLO
#include <armadillo>
#endif  // USE_RCPPARMADILLO

#endif  // ARMADILLO_CONFIG
