#include "GWPCA.h"

using namespace arma;
using namespace gwm;

void GWPCA::run()
{
    GWM_LOG_STAGE("Initialization");
    createDistanceParameter();
    GWM_LOG_STOP_RETURN(mStatus, void());

    GWM_LOG_STAGE("Solving");
    mLocalPV = pca(mX, mLoadings, mScores, mSDev);
    GWM_LOG_STOP_RETURN(mStatus, void());
    
    mWinner = index_max(mLoadings.slice(0), 1);
}

mat GWPCA::solveSerial(const mat& x, cube& loadings, cube& scores, mat& sdev)
{
    uword nDp = mCoords.n_rows, nVar = mX.n_cols;
    mat d_all(nVar, nDp, arma::fill::zeros);
    vec w0;
    loadings = cube(nDp, nVar, mK, arma::fill::zeros);
    scores = cube(nDp, nDp, mK, arma::fill::zeros);
    for (uword i = 0; i < nDp; i++)
    {
        GWM_LOG_STOP_BREAK(mStatus);
        vec w = mSpatialWeight.weightVector(i);
        mat U, V;
        vec d;
        wpca(x, w, U, V, d);
        w0 = w;
        d_all.col(i) = d;
        for (int j = 0; j < mK; j++)
        {
            loadings.slice(j).row(i) = arma::trans(V.col(j));
            scores.slice(j).col(i) = U.col(j);
        }
        GWM_LOG_PROGRESS(i + 1, nDp);
    }
    d_all = trans(d_all);
    mat variance = (d_all / sqrt(sum(w0))) % (d_all / sqrt(sum(w0)));
    sdev = sqrt(variance);
    mat pv = variance.cols(0, mK - 1).each_col() % (1.0 / sum(variance, 1)) * 100.0;
    return pv;
}

void GWPCA::wpca(const mat& x, const vec& w, mat& U, mat& V, vec & d)
{
    mat xw = x.each_col() % w;
    mat centerized = (x.each_row() - sum(xw) / sum(w)).each_col() % sqrt(w);
    svd(U, d, V, centerized);
}

void GWPCA::rwpca(const mat& x, const vec& w, mat& U, mat& V, vec & d)
{
    mat mids = x;
    uword medianIdx = (abs(w - 0.5)).index_min();
    mids = mids.each_row() - x.row(medianIdx);
    mat weighted = mids.each_col() % w;
    mat score;
    vec tsquared;
    princomp(V, score, d, tsquared, weighted);
    U = score;
}

mat GWPCA::solveRobustSerial(const mat& x, cube& loadings, cube& scores, mat& sdev)
{
    uword nDp = mCoords.n_rows, nVar = mX.n_cols;
    mat d_all(nVar, nDp, arma::fill::zeros);
    vec w0;
    loadings = cube(nDp, nVar, mK, arma::fill::zeros);
    scores = cube(nDp, nDp, mK, arma::fill::zeros);
    for (uword i = 0; i < nDp; i++)
    {
        GWM_LOG_STOP_BREAK(mStatus);
        vec w = mSpatialWeight.weightVector(i);
        uvec positive = find(w > 0);
        vec newWt = w.elem(positive);
        mat newX = x.rows(positive);
        if (newWt.n_rows <= 5)
        {
            continue;
        }
        mat U, V;
        vec d;
        rwpca(newX, newWt, U, V, d);
        w0 = newWt;
        d_all.col(i) = d;
        for (int j = 0; j < mK; j++)
        {
            loadings.slice(j).row(i) = arma::trans(V.col(j));
            mat scoreAll = x.each_row() % arma::trans(V.col(j));
            scores.slice(j).col(i) = sum(scoreAll, 1);
        }
        GWM_LOG_PROGRESS(i + 1, nDp);
    }
    d_all = trans(d_all);
    mat variance = (d_all / pow(sum(w0), 0.5)) % (d_all / pow(sum(w0), 0.5));
    sdev = sqrt(variance);
    mat pv = variance.cols(0, mK - 1).each_col() % (1.0 / sum(variance, 1)) * 100.0;
    return pv;
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
