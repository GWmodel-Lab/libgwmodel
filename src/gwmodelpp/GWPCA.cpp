#include "GWPCA.h"

using namespace arma;
using namespace gwm;

void GWPCA::run()
{
    createDistanceParameter();
    GWM_LOG_STOP_RETURN(mStatus, void());

    mLocalPV = pca(mX, mLoadings, mSDev);
    GWM_LOG_STOP_RETURN(mStatus, void());
    
    mWinner = index_max(mLoadings.slice(0), 1);
}

mat GWPCA::solveSerial(const mat& x, cube& loadings, mat& sdev)
{
    uword nDp = mCoords.n_rows, nVar = mX.n_cols;
    mat d_all(nVar, nDp, arma::fill::zeros);
    vec w0;
    loadings = cube(nDp, nVar, mK, arma::fill::zeros);
    for (uword i = 0; i < nDp; i++)
    {
        GWM_LOG_STOP_BREAK(mStatus);
        vec w = mSpatialWeight.weightVector(i);
        mat V;
        vec d;
        wpca(x, w, V, d);
        w0 = w;
        d_all.col(i) = d;
        for (int j = 0; j < mK; j++)
        {
            loadings.slice(j).row(i) = arma::trans(V.col(j));
        }
        GWM_LOG_PROGRESS(i + 1, nDp);
    }
    d_all = trans(d_all);
    mat variance = (d_all / sqrt(sum(w0))) % (d_all / sqrt(sum(w0)));
    sdev = sqrt(variance);
    mat pv = variance.cols(0, mK - 1).each_col() % (1.0 / sum(variance, 1)) * 100.0;
    return pv;
}

void GWPCA::wpca(const mat& x, const vec& w, mat& V, vec & d)
{
    mat xw = x.each_col() % w, U;
    mat centerized = (x.each_row() - sum(xw) / sum(w)).each_col() % sqrt(w);
    svd(U, d, V, centerized);
}

bool GWPCA::isValid()
{
    if (SpatialAlgorithm::isValid())
    {
        if (mK > 0)
        {
            return true;
        }
        else return false;
    }
    else return false;
}
