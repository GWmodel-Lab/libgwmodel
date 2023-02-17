#include "GWDA.h"
#include <assert.h>
#include <vector>

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

using namespace arma;
using namespace gwm;
using namespace std;

vec GWDA::del(vec x, uword rowcount)
{
    vec res;
    if (rowcount == 0)
        res = x.rows(rowcount + 1, x.n_rows - 1);
    else if (rowcount == x.n_rows - 1)
        res = x.rows(0, x.n_rows - 2);
    else
        res = join_cols(x.rows(0, rowcount - 1), x.rows(rowcount + 1, x.n_rows - 1));
    return res;
}

vec GWDA::findq(const mat &x, const vec &w)
{
    uword lw = w.n_rows;
    uword lp = 3;
    vec q = vec(lp, fill::zeros);
    vec xo = sort(x);
    vec wo = w(sort_index(x));
    vec Cum = cumsum(wo);
    uword cond = lw - 1;
    for (uword j = 0; j < lp; j++)
    {
        double k = 0.25 * (j + 1);
        for (uword i = 0; i < lw; i++)
        {
            if (Cum(i) > k)
            {
                cond = i - 1;
                break;
            }
        }
        if (cond < 0)
        {
            cond = 0;
        }
        q.row(j) = xo[cond];
        cond = lw - 1;
    }
    return q;
}

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
    uword nRp = mCoords.n_rows;// nVar = mX.n_cols, nPr = mprX.n_rows;
    vec lev = levels(mY);
    mat wt;
    for (uword i = 0; i < nRp; i++)
    {
        wt.row(i) = mSpatialWeight.weightVector(i);
    }
    if (misWqda)
    {
        mat res = wqda(mX, mY, wt, mprX, mHascov, mHasmean, mHasprior);
    }
    else
    {
        mat res = wlda(mX, mY, wt, mprX, mHascov, mHasmean, mHasprior);
    }
}

vector<mat> GWDA::splitX(arma::mat &x, arma::vec &y)
{
    vec lev = levels(y);
    uword p = lev.size();
    vector<mat> res;
    for (uword i = 0; i < p; i++)
    {
        res[lev[i]] = x.rows(find(y == lev[i]));
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
        if (any(find(y(i) == lev))==true)
        {
            continue;
        }
        else
        {
            lev(index++) = y(i);
        }
    }
    return lev;
}

mat GWDA::covwtmat(const arma::mat &x, const arma::vec &wt)
{
    uword n = x.n_rows;
    mat sigma(n, n, fill::zeros);
    double w = sum(wt) / (sum(wt) * sum(wt) - sum(wt * wt));
    vec average;
    for (uword i = 0; i < n; i++)
    {
        average(i) = sum(x.row(i) % wt.row(i)) / sum(wt.row(i));
    }
    for (uword j = 0; j < n; j++)
    {
        for (uword k = 0; k < n; k++)
        {
            sigma(j, k) = w * sum(wt * (x.col(j) - average(j)) * (x.col(k) - average(k)));
        }
    }
    return sigma;
}

mat GWDA::wqda(arma::mat &x, arma::vec &y, arma::mat &wt, arma::mat &xpr, bool hasCOv, bool hasMean, bool hasPrior)
{
    vec lev = levels(y);
    uword m = lev.n_rows;
    uword nDp = x.n_rows;
    uword nPr = xpr.n_rows;
    mat wtOnes;
    wtOnes.ones(nDp, nPr);
    vector<mat> xg = splitX(x, y);
    vector<vec> prior;
    /*for(uword i=0;i<m;i++){
        prior.row(i)=vec(m,fill::zeros);
    }*/
    mat wti;
    vector<mat> localMean;
    vector<cube> sigmagw;
    double sumW;
    for (uword i = 0; i < m; i++)
    {
        mat xi = xg[lev(i)];
        vec idx;
        uword index = 0;
        for (uword j = 0; j < y.size(); j++)
        {
            if (y[j] == lev[i])
            {
                idx[index++] = j;
            }
        }
        // uword idx=xi;
        if (hasMean)
        {
            for (uword j = 0; j < idx.size(); j++)
            {
                wti.row(idx[j]) = mSpatialWeight.weightVector(j);
            }
        }
        else
        {
            wti = wtOnes.rows(find(y == lev[i]));
        }
        localMean[lev[i]] = wMean(xi, wti);
        if (hasCOv)
        {
            for (uword j = 0; j < idx.size(); j++)
            {
                wti.row(idx[j]) = mSpatialWeight.weightVector(j);
            }
        }
        else
        {
            wti = wtOnes.rows(find(y == lev[i]));
        }
        sigmagw[lev(i)] = wVarCov(xi, wti);
        if (hasPrior)
        {
            for (uword j = 0; j < idx.size(); j++)
            {
                wti.row(idx[j]) = mSpatialWeight.weightVector(j);
            }
            sumW = accu(wt);
        }
        else
        {
            wti = wtOnes.rows(find(y == lev[i]));
            sumW = accu(wtOnes);
        }
        prior[lev[i]] < -wPrior(wti, sumW);
    }
    mat logPf = mat(nPr, m, fill::zeros);
    for (uword i = 0; i < m; i++)
    {
        for (uword j = 0; i < nPr; j++)
        {
            vec xprj = xpr.row(j);
            vec meani = localMean[lev[i]].row(j);
            mat covmatj = sigmagw[lev(i)].row(j);
            vec x1 = 0.5 * (xprj - meani).t() * inv(covmatj) * (xprj - meani);
            logPf(j, i) = (m / 2) * log(norm(covmatj)) + x1(0) - log(prior[lev(i)].at(j));
        }
    }
    return logPf;
    /*colnames(log.pf) <- paste(lev, "logp", sep="_");
        group.pr <- vector("character", pr.n)
        for(i in 1:pr.n)
            group.pr[i] <- lev[which.min(log.pf[i,])[1]]
        res.df <- cbind(log.pf, group.pr)
        colnames(res.df) <-c(colnames(log.pf), "group.predicted")
        res.df */
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
    /*for(uword i=0;i<m;i++){
        prior.row(i)=vec(m,fill::zeros);
    }*/
    mat wti;
    vector<mat> localMean;
    vector<cube> sigmagw;
    double sumW;
    for (uword i = 0; i < m; i++)
    {
        mat xi = xg[lev(i)];
        vec idx;
        uword index = 0;
        for (uword j = 0; j < y.size(); j++)
        {
            if (y[j] == lev[i])
            {
                idx[index++] = j;
            }
        }
        // uword idx=xi;
        if (hasMean)
        {
            for (uword j = 0; j < idx.size(); j++)
            {
                wti.row(idx[j]) = mSpatialWeight.weightVector(j);
            }
        }
        else
        {
            wti = wtOnes.rows(find(y == lev[i]));
        }
        localMean[lev[i]] = wMean(xi, wti);
        if (hasCOv)
        {
            for (uword j = 0; j < idx.size(); j++)
            {
                wti.row(idx[j]) = mSpatialWeight.weightVector(j);
            }
        }
        else
        {
            wti = wtOnes.rows(find(y == lev[i]));
        }
        sigmagw[lev(i)] = wVarCov(xi, wti);
        if (hasPrior)
        {
            for (uword j = 0; j < idx.size(); j++)
            {
                wti.row(idx[j]) = mSpatialWeight.weightVector(j);
            }
            sumW = accu(wt);
        }
        else
        {
            wti = wtOnes.rows(find(y == lev[i]));
            sumW = accu(wtOnes);
        }
        prior[lev[i]] < -wPrior(wti, sumW);
    }
    cube sigma1gw(nPr, nVar, nVar, fill::zeros);
    vec counts = y;
    for (uword i = 0; i < nPr; i++)
    {
        mat sigmai(nVar, nVar, fill::zeros);
        for (uword j = 0; j < m; j++)
        {
            double yi = counts(j);
            mat x1 = sigmagw[lev(i)].row(i);
            sigmai = sigmai + yi * x1;
        }
        sigma1gw.row(i) = sigmai / sum(counts);
    }

    mat logPf = mat(nPr, m, fill::zeros);
    for (uword i = 0; i < m; i++)
    {
        for (uword j = 0; i < nPr; j++)
        {
            vec xprj = xpr.row(j);
            vec meani = localMean[lev[i]].row(j);
            mat covmatj = sigmagw[lev(i)].row(j);
            vec x1 = 0.5 * (xprj - meani).t() * inv(covmatj) * (xprj - meani);
            logPf(j, i) = (m / 2) * log(norm(covmatj)) + x1(0) - log(prior[lev(i)].at(j));
        }
    }
    return logPf;
    /*colnames(log.pf) <- paste(lev, "logp", sep="_");
        group.pr <- vector("character", pr.n)
        for(i in 1:pr.n)
            group.pr[i] <- lev[which.min(log.pf[i,])[1]]
        res.df <- cbind(log.pf, group.pr)
        colnames(res.df) <-c(colnames(log.pf), "group.predicted")
        res.df */
}

mat GWDA::wMean(arma::mat &x, arma::mat &wt)
{
    uword nVar = x.n_cols;
    // uword nRp = x.n_rows;
    uword nPr = wt.n_cols;
    mat localMean = mat(nPr, nVar, fill::zeros);
    for (uword i = 0; i < nPr; i++)
    {
        vec w = wt.col(i).t();
        double sumw = sum(w);
        vec Wi = w / sumw;
        localMean.row(i) = trans(Wi) * x;
    }
    return localMean;
}

cube GWDA::wVarCov(arma::mat &x, arma::mat &wt)
{
    uword nVar = x.n_cols;
    // uword nRp=x.n_rows;
    uword nPr = wt.n_cols;
    // uword corrSize = mIsCorrWithFirstOnly ? 1 : nVar - 1;
    cube Covmat(nPr, nVar, nVar, fill::zeros);
    for (uword i = 0; i < nPr; i++)
    {
        vec w = wt.col(i);
        double sumw = sum(w);
        vec wi = w / sumw;
        if (nVar >= 2)
        {
            /*uword tag = 0;
            for (uword j = 0; j < corrSize; j++)
            {
                for (uword k = j + 1; k < nVar; k++)
                {
                    double covjk = covwt(x.col(j), x.col(k), wi);
                    Covmat.row(i) = covwt(x,wi).;
                    tag++;
                }
            }*/
            Covmat.row(i) = covwtmat(x, wi);
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

void GWDA::summarySerial()
{
    mat rankX = mX;
    rankX.each_col([&](vec &x)
                   { x = rank(x); });
    uword nVar = mX.n_cols, nRp = mCoords.n_rows;
    uword corrSize = mIsCorrWithFirstOnly ? 1 : nVar - 1;
    for (uword i = 0; i < nRp; i++)
    {
        vec w = mSpatialWeight.weightVector(i);
        double sumw = sum(w);
        vec Wi = w / sumw;
        mLocalMean.row(i) = trans(Wi) * mX;
        if (mQuantile)
        {
            mat quant = mat(3, nVar);
            for (uword j = 0; j < nVar; j++)
            {
                quant.col(j) = findq(mX.col(j), Wi);
            }
            mLocalMedian.row(i) = quant.row(1);
            mIQR.row(i) = quant.row(2) - quant.row(0);
            mQI.row(i) = (2 * quant.row(1) - quant.row(2) - quant.row(0)) / mIQR.row(i);
        }
        mat centerized = mX.each_row() - mLocalMean.row(i);
        mLVar.row(i) = Wi.t() * (centerized % centerized);
        mStandardDev.row(i) = sqrt(mLVar.row(i));
        mLocalSkewness.row(i) = (Wi.t() * (centerized % centerized % centerized)) / (mLVar.row(i) % mStandardDev.row(i));
        if (nVar >= 2)
        {
            uword tag = 0;
            for (uword j = 0; j < corrSize; j++)
            {
                for (uword k = j + 1; k < nVar; k++)
                {
                    double covjk = covwt(mX.col(j), mX.col(k), Wi);
                    double sumW2 = sum(Wi % Wi);
                    double covjj = mLVar(i, j) / (1.0 - sumW2);
                    double covkk = mLVar(i, k) / (1.0 - sumW2);
                    mCovmat(i, tag) = covjk;
                    mCorrmat(i, tag) = covjk / sqrt(covjj * covkk);
                    mSCorrmat(i, tag) = corwt(rankX.col(j), rankX.col(k), Wi);
                    tag++;
                }
            }
        }
    }
    mLCV = mStandardDev / mLocalMean;
}

#ifdef ENABLE_OPENMP
void GWDA::summaryOmp()
{
    mat rankX = mX;
    rankX.each_col([&](vec &x)
                   { x = rank(x); });
    uword nVar = mX.n_cols, nRp = mCoords.n_rows;
    uword corrSize = mIsCorrWithFirstOnly ? 1 : nVar - 1;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; (uword)i < nRp; i++)
    {
        vec w = mSpatialWeight.weightVector(i);
        double sumw = sum(w);
        vec Wi = w / sumw;
        mLocalMean.row(i) = trans(Wi) * mX;
        if (mQuantile)
        {
            mat quant = mat(3, nVar);
            for (uword j = 0; j < nVar; j++)
            {
                quant.col(j) = findq(mX.col(j), Wi);
            }
            mLocalMedian.row(i) = quant.row(1);
            mIQR.row(i) = quant.row(2) - quant.row(0);
            mQI.row(i) = (2 * quant.row(1) - quant.row(2) - quant.row(0)) / mIQR.row(i);
        }
        mat centerized = mX.each_row() - mLocalMean.row(i);
        mLVar.row(i) = Wi.t() * (centerized % centerized);
        mStandardDev.row(i) = sqrt(mLVar.row(i));
        mLocalSkewness.row(i) = (Wi.t() * (centerized % centerized % centerized)) / (mLVar.row(i) % mStandardDev.row(i));
        if (nVar >= 2)
        {
            uword tag = 0;
            for (uword j = 0; j < corrSize; j++)
            {
                for (uword k = j + 1; k < nVar; k++)
                {
                    double covjk = covwt(mX.col(j), mX.col(k), Wi);
                    double sumW2 = sum(Wi % Wi);
                    double covjj = mLVar(i, j) / (1.0 - sumW2);
                    double covkk = mLVar(i, k) / (1.0 - sumW2);
                    mCovmat(i, tag) = covjk;
                    mCorrmat(i, tag) = covjk / sqrt(covjj * covkk);
                    mSCorrmat(i, tag) = corwt(rankX.col(j), rankX.col(k), Wi);
                    tag++;
                }
            }
        }
    }
    mLCV = mStandardDev / mLocalMean;
}
#endif

void GWDA::setParallelType(const ParallelType &type)
{
    if (type & parallelAbility())
    {
        mParallelType = type;
        switch (type)
        {
        case ParallelType::SerialOnly:
            mSummaryFunction = &GWDA::summarySerial;
            break;
#ifdef ENABLE_OPENMP
        case ParallelType::OpenMP:
            mSummaryFunction = &GWDA::summaryOmp;
            break;
#endif
        default:
            mSummaryFunction = &GWDA::summarySerial;
            break;
        }
    }
}