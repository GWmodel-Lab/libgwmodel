#include "CGwmGWPCA.h"

void CGwmGWPCA::run()
{
    createDistanceParameter();
    mLocalPV = pca(mX, mLoadings, mSDev);
    mWinner = index_max(mLoadings.slice(0), 1);
}

mat CGwmGWPCA::solveSerial(const mat& x, cube& loadings, mat& sdev)
{
    uword nDp = mCoords.n_rows, nVar = mX.n_cols;
    mat d_all(nVar, nDp, arma::fill::zeros);
    vec w0;
    loadings = cube(nDp, nVar, mK, arma::fill::zeros);
    for (uword i = 0; i < nDp; i++)
    {
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
    }
    d_all = trans(d_all);
    mat variance = (d_all / sqrt(sum(w0))) % (d_all / sqrt(sum(w0)));
    sdev = sqrt(variance);
    mat pv = variance.cols(0, mK - 1).each_col() % (1.0 / sum(variance, 1)) * 100.0;
    return pv;
}

void CGwmGWPCA::wpca(const mat& x, const vec& w, mat& V, vec & d)
{
    mat xw = x.each_col() % w, U;
    mat centerized = (x.each_row() - sum(xw) / sum(w)).each_col() % sqrt(w);
    svd(U, d, V, centerized);
}

bool CGwmGWPCA::isValid()
{
    if (CGwmSpatialAlgorithm::isValid())
    {
        if (mK > 0)
        {
            return true;
        }
        else return false;
    }
    else return false;
}
