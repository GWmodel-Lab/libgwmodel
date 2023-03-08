#include "GWDA.h"
#include <assert.h>
#include <vector>

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

using namespace arma;
using namespace gwm;
using namespace std;

bool GWDA::isValid()
{
    if (SpatialAlgorithm::isValid())
    {
        if (!(mX.n_cols > 0))
            return false;

        return true;
    }
    else
        return false;
}

void GWDA::run()
{
    createDistanceParameter();
    uword nRp=mCoords.n_rows;
    //vec lev = levels(mY);
    if (mprX.is_empty())
    {
        mprX = mX;
    }
    if(mprY.is_empty()){
        mprY=mY;
    }
    (this->*mDiscriminantAnalysisFunction)();
    uword nCol=mRes.n_cols;
    uvec correct=find((mRes.col(nCol-1)==mY)==1);
    mCorrectRate=(double)correct.n_rows/nRp;
}

void GWDA::discriminantAnalysisSerial()
{
    uword nRp = mCoords.n_rows;
    // uword nVar = mX.n_cols;//nPr = mprX.n_rows;
    vec lev = levels(mY);
    mat wt(nRp, nRp, fill::zeros);
    for (uword i = 0; i < nRp; i++)
    {
        // vec w=mSpatialWeight.weightVector(i);
        wt.row(i) = mSpatialWeight.weightVector(i).t();
    }
    if (mIsWqda)
    {
        mRes = wqda(mX, mY, wt, mprX, mHascov, mHasmean, mHasprior);
    }
    else
    {
        mRes = wlda(mX, mY, wt, mprX, mHascov, mHasmean, mHasprior);
    }
}

#ifdef ENABLE_OPENMP
void GWDA::discriminantAnalysisOmp()
{
    uword nRp = mCoords.n_rows;
    // uword nVar = mX.n_cols;//nPr = mprX.n_rows;
    vec lev = levels(mY);
    mat wt(nRp, nRp, fill::zeros);
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (uword i = 0; i < nRp; i++)
    {
        wt.col(i) = mSpatialWeight.weightVector(i);
    }
    if (mIsWqda)
    {
        mRes = wqda(mX, mY, wt, mprX, mHascov, mHasmean, mHasprior);
    }
    else
    {
        mRes = wlda(mX, mY, wt, mprX, mHascov, mHasmean, mHasprior);
    }
}
#endif

vector<mat> GWDA::splitX(arma::mat &x, arma::vec &y)
{
    vec lev = levels(y);
    uword p = lev.size();
    vector<mat> res;
    for (uword i = 0; i < p; i++)
    {
        res.push_back(x.rows(find(y == lev[i])));
    }
    return res;
}

vec GWDA::levels(arma::vec &y)
{
    uword n = y.size();
    vec lev;
    uword index = 0;
    for (uword i = 0; i < n; i++)
    {
        if (any(lev == y(i)) == false)
        {
            lev.resize(index + 1);
            lev(index) = y(i);
            index++;
        }
    }
    return lev;
}

mat GWDA::covwtmat(const arma::mat &x, const arma::vec &wt)
{
    uword n = x.n_cols;
    mat sigma(n, n, fill::zeros);
    double w = sum(wt) / (sum(wt) * sum(wt) - sum(wt % wt));
    vec average(n,fill::zeros);
    for (uword i = 0; i < n; i++)
    {
        average(i) = sum(x.col(i) % wt) / sum(wt);
    }
    for (uword j = 0; j < n; j++)
    {
        for (uword k = 0; k < n; k++)
        {
            sigma(j, k) = w * sum(wt % ((x.col(j) - average(j)) % (x.col(k) - average(k))));
        }
    }
    return sigma;
}

mat GWDA::wqda(arma::mat &x, arma::vec &y, arma::mat &wt, arma::mat &xpr, bool hasCOv, bool hasMean, bool hasPrior)
{
    vec lev = levels(y);
    uword m = lev.n_rows;
    // uword nDp = x.n_rows;
    uword nPr = xpr.n_rows;
    mat wtOnes;
    // wtOnes.ones(nDp, nPr);
    vector<mat> xg = splitX(x, y);
    vector<vec> prior;
    mat wti; //(nDp, nPr, fill::zeros);
    vector<mat> localMean;
    vector<cube> sigmagw;
    double sumW;
    for (uword i = 0; i < m; i++)
    {
        mat xi = xg[i];
        /*vector<uword> idx;
        // uword index = 0;
        for (uword j = 0; j < y.size(); j++)
        {
            if (y[j] == lev[i])
            {
                idx.push_back(j);
            }
        }*/
        if (hasMean)
        {
            /*for (uword j = 0; j < idx.size(); j++)
            {
                wti.row(idx[j]) = mSpatialWeight.weightVector(j);
            }*/
            wti = wt.rows(find(y == lev[i]));
        }
        else
        {
            wti = wtOnes.rows(find(y == lev[i]));
        }
        localMean.push_back(wMean(xi, wti));
        if (hasCOv)
        {
            wti = wt.rows(find(y == lev[i]));
        }
        else
        {
            wti = wtOnes.rows(find(y == lev[i]));
        }
        sigmagw.push_back(wVarCov(xi, wti));
        if (hasPrior)
        {
            wti = wt.rows(find(y == lev[i]));
            sumW = accu(wt);
        }
        else
        {
            wti = wtOnes.rows(find(y == lev[i]));
            sumW = accu(wtOnes);
        }
        prior.push_back(wPrior(wti, sumW));
    }
    mat logPf = mat(nPr, m, fill::zeros);
    for (uword i = 0; i < m; i++)
    {
        for (uword j = 0; j < nPr; j++)
        {
            vec xprj = xpr.row(j).t();
            vec meani = localMean[i].row(j).t();
            mat covmatj = sigmagw[i].row(j);
            vec x1 = 0.5 * (xprj - meani).t() * solve(covmatj,(xprj-meani));//inv(covmatj) * (xprj - meani);
            logPf(j, i) = (m / 2) * log(norm(covmatj)) + x1(0) - log(prior[i].at(j));
        }
    }
    mat groupPr=mat(nPr,1,fill::zeros);
    for (uword i = 0; i < nPr; i++)
    {
        uvec index = find(logPf.row(i) == min(logPf.row(i)));
        groupPr(i, 0) = lev(index(0));
    }
    logPf=join_rows(logPf,groupPr);
    return logPf;
}

mat GWDA::wlda(arma::mat &x, arma::vec &y, arma::mat &wt, arma::mat &xpr, bool hasCOv, bool hasMean, bool hasPrior)
{
    vec lev = levels(y);
    uword m = lev.n_rows;
    uword nDp = x.n_rows;
    uword nPr = xpr.n_rows;
    uword nVar = x.n_cols;
    mat wtOnes;
    wtOnes.ones(nDp, nPr);
    vector<mat> xg = splitX(x, y);
    vector<vec> prior;
    mat wti(nDp, nPr, fill::zeros); 
    vector<mat> localMean;
    vector<cube> sigmagw;
    double sumW;
    for (uword i = 0; i < m; i++)
    {
        mat xi = xg[i];
        /*vector<uword> idx;
        // uword index = 0;
        for (uword j = 0; j < y.size(); j++)
        {
            if (y[j] == lev[i])
            {
                idx.push_back(j);
            }
        }*/
        if (hasMean)
        {
            wti = wt.rows(find(y == lev[i]));
        }
        else
        {
            wti = wtOnes.rows(find(y == lev[i]));
        }
        localMean.push_back(wMean(xi, wti));
        if (hasCOv)
        {
            wti = wt.rows(find(y == lev[i]));
        }
        else
        {
            wti = wtOnes.rows(find(y == lev[i]));
        }
        sigmagw.push_back(wVarCov(xi, wti));
        if (hasPrior)
        {
            wti = wt.rows(find(y == lev[i]));
            sumW = accu(wt);
        }
        else
        {
            wti = wtOnes.rows(find(y == lev[i]));
            sumW = accu(wtOnes);
        }
        prior.push_back(wPrior(wti, sumW));
    }
    cube sigma1gw(nPr, nVar, nVar, fill::zeros);
    vec counts = y;
    for (uword i = 0; i < nPr; i++)
    {
        mat sigmai(nVar, nVar, fill::zeros);
        for (uword j = 0; j < m; j++)
        {
            double yi = counts(j);
            mat x1 = sigmagw[j].row(i);
            sigmai = sigmai + yi * x1;
        }
        sigma1gw.row(i) = sigmai / sum(counts);
    }

    mat logPf = mat(nPr, m, fill::zeros);
    for (uword i = 0; i < m; i++)
    {
        for (uword j = 0; j < nPr; j++)
        {
            vec xprj = xpr.row(j).t();
            vec meani = localMean[i].row(j).t();
            mat covmatj = sigma1gw.row(j);
            vec x1 = 0.5 * (xprj - meani).t() * solve(covmatj,(xprj-meani));
            logPf(j, i) = (m / 2) * log(norm(covmatj)) + x1(0) - log(prior[i].at(j));
        }
    }
    mat groupPr=mat(nPr,1,fill::zeros);
    for (uword i = 0; i < nPr; i++)
    {
        uvec index = find(logPf.row(i) == min(logPf.row(i)));
        groupPr(i, 0) = lev(index(0));
    }
    logPf=join_rows(logPf,groupPr);
    return logPf;
}

mat GWDA::wMean(arma::mat &x, arma::mat &wt)
{
    uword nVar = x.n_cols;
    uword nPr = wt.n_cols;
    mat localMean = mat(nPr, nVar, fill::zeros);
    for (uword i = 0; i < nPr; i++)
    {
        rowvec w = wt.col(i).t();
        double sumw = sum(w);
        rowvec wi = w / sumw;
        localMean.row(i) = wi * x;
    }
    return localMean;
}

cube GWDA::wVarCov(arma::mat &x, arma::mat &wt)
{
    uword nVar = x.n_cols;
    uword nPr = wt.n_cols;
    cube Covmat(nPr, nVar, nVar, fill::zeros);
    for (uword i = 0; i < nPr; i++)
    {
        vec w = wt.col(i);
        double sumw = sum(w);
        vec wi = w / sumw;
        if (nVar >= 2)
        {
            Covmat.row(i) = covwtmat(x, wi);
        }else{

        }
    }
    return Covmat;
}

vec GWDA::wPrior(arma::mat &wt, double sumW)
{
    uword nPr = wt.n_cols;
    vec localPrior = vec(nPr, fill::zeros);
    for (uword i = 0; i < nPr; i++)
    {
        localPrior(i) = sum(wt.col(i)) / sumW;
    }
    return localPrior;
}

void GWDA::setParallelType(const ParallelType &type)
{
    if (type & parallelAbility())
    {
        mParallelType = type;
        switch (type)
        {
        case ParallelType::SerialOnly:
            mDiscriminantAnalysisFunction = &GWDA::discriminantAnalysisSerial;
            break;
#ifdef ENABLE_OPENMP
        case ParallelType::OpenMP:
            mDiscriminantAnalysisFunction = &GWDA::discriminantAnalysisOmp;
            break;
#endif
        default:
            mDiscriminantAnalysisFunction = &GWDA::discriminantAnalysisSerial;
            break;
        }
    }
}