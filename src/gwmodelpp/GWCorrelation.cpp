#include "GWCorrelation.h"
#include <assert.h>
#include <map>

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace arma;
using namespace gwm;