#include <armadillo>
#include <gwmodel.h>
using namespace gwm;
using namespace arma;

int main()
{
    mat coords(100, 2, fill::randu);
    mat x = mat(100, 3, fill::randu);
    BandwidthWeight bw(36.0, true, BandwidthWeight::Gaussian);
    CRSDistance dist(false);
    SpatialWeight sw(&bw, &dist);
    GWPCA algorithm(x, coords, sw);
    algorithm.setKeepComponents(2);
    algorithm.run();
    return 0;
}
