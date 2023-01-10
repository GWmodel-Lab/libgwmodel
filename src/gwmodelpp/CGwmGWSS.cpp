#include "CGwmGWSS.h"
#include <assert.h>

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

using namespace arma;
using namespace gwm;

vec CGwmGWSS::del(vec x, uword rowcount){
    vec res;
    if (rowcount == 0)
        res = x.rows(rowcount + 1, x.n_rows - 1);
    else if (rowcount == x.n_rows - 1)
        res = x.rows(0,x.n_rows-2);
    else
        res = join_cols(x.rows(0,rowcount - 1),x.rows(rowcount + 1, x.n_rows - 1));
    return res;
}

vec CGwmGWSS::findq(const mat &x, const vec &w)
{
    uword lw = w.n_rows;
    uword lp = 3;
    vec q = vec(lp,fill::zeros);
    vec xo = sort(x);
    vec wo = w(sort_index(x));
    vec Cum = cumsum(wo);
    uword cond = lw - 1;
    for(uword j = 0; j < lp ; j++){
        double k = 0.25 * (j + 1);
        for(uword i = 0; i < lw; i++){
            if(Cum(i) > k){
                cond = i - 1;
                break;
            }
        }
        if(cond < 0)
        {
            cond = 0;
        }
        q.row(j) = xo[cond];
        cond = lw - 1;
    }
    return q;
}

bool CGwmGWSS::isValid()
{
    if (CGwmSpatialMonoscaleAlgorithm::isValid())
    {
        if (!(mX.n_cols > 0))
            return false;

        return true;
    }
    else return false;
}

void CGwmGWSS::run()
{
    createDistanceParameter();
    uword nRp = mCoords.n_rows, nVar = mX.n_cols;
    mLocalMean = mat(nRp, nVar, fill::zeros);
    mStandardDev = mat(nRp, nVar, fill::zeros);
    mLocalSkewness = mat(nRp, nVar, fill::zeros);
    mLCV = mat(nRp, nVar, fill::zeros);
    mLVar = mat(nRp, nVar, fill::zeros);
    if (mQuantile)
    {
        mLocalMedian = mat(nRp, nVar, fill::zeros);
        mIQR = mat(nRp, nVar, fill::zeros);
        mQI = mat(nRp, nVar, fill::zeros);
    }
    if (nVar > 1)
    {
        uword nCol = mIsCorrWithFirstOnly ? (nVar - 1) : (nVar + 1) * nVar / 2 - nVar;
        mCovmat = mat(nRp, nCol, fill::zeros);
        mCorrmat = mat(nRp, nCol, fill::zeros);
        mSCorrmat = mat(nRp, nCol, fill::zeros);
    }
    (this->*mSummaryFunction)();
}

void CGwmGWSS::summarySerial()
{
    mat rankX = mX;
    rankX.each_col([&](vec& x) { x = rank(x); });
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
void CGwmGWSS::summaryOmp()
{
    mat rankX = mX;
    rankX.each_col([&](vec& x) { x = rank(x); });
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

void CGwmGWSS::setParallelType(const ParallelType &type)
{
    if (type & parallelAbility())
    {
        mParallelType = type;
        switch (type) {
        case ParallelType::SerialOnly:
            mSummaryFunction = &CGwmGWSS::summarySerial;
            break;
#ifdef ENABLE_OPENMP
        case ParallelType::OpenMP:
            mSummaryFunction = &CGwmGWSS::summaryOmp;
            break;
#endif
        default:
            mSummaryFunction = &CGwmGWSS::summarySerial;
            break;
        }
    }
}