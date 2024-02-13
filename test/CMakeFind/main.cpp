#include <armadillo>
#include <gwmodel.h>
using namespace gwm;
using namespace arma;

int main()
{
    mat coords(100, 2, fill::randu);
    mat x = join_rows(vec(100, fill::ones), mat(100, 2, fill::randu));
    mat betas = mat(100, 3, fill::randu);
    vec eps(100, fill::randu);
    vec y = sum(x % betas, 1) + eps;
    BandwidthWeight bw(36.0, true, BandwidthWeight::Gaussian);
    CRSDistance dist(false);
    SpatialWeight sw(&bw, &dist);
    GWRBasic algorithm(x, y, coords, sw);
    algorithm.fit();
    return 0;
}
