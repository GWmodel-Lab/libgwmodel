#include "GWCorrelation.h"
#include "BandwidthSelector.h"
#include <assert.h>
#include "Logger.h"
#include <map>

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace arma;
using namespace gwm;

vec GWCorrelation::del(vec x, uword rowcount){
    vec res;
    if (rowcount == 0)
        res = x.rows(rowcount + 1, x.n_rows - 1);
    else if (rowcount == x.n_rows - 1)
        res = x.rows(0,x.n_rows-2);
    else
        res = join_cols(x.rows(0,rowcount - 1),x.rows(rowcount + 1, x.n_rows - 1));
    return res;
}

vec GWCorrelation::findq(const mat &x, const vec &w)
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

void GWCorrelation::run()
{
    GWM_LOG_STAGE("Initializing");
    uword nDp = mCoords.n_rows, nVar = mX.n_cols, nRsp=mY.n_cols;
    createDistanceParameter(nVar);
    GWM_LOG_STOP_RETURN(mStatus, void());

    uword nCol = nVar * nRsp;
    mLVar = mat(nDp, nVar, fill::zeros);
    mLocalMean = mat(nDp, nVar, fill::zeros);
    mCovmat = mat(nDp, nCol, fill::zeros);
    mCorrmat = mat(nDp, nCol, fill::zeros);
    mSCorrmat = mat(nDp, nCol, fill::zeros);
    if (mQuantile)
    {
        mLocalMedian = mat(nDp, nVar, fill::zeros);
        mIQR = mat(nDp, nVar, fill::zeros);
        mQI = mat(nDp, nVar, fill::zeros);
    }

    GWM_LOG_STAGE("Initializing bandwidths");
    for (uword i = 0; i < nCol; i++)
    {
        if(mBandwidthInitilize[i] == BandwidthInitilizeType::Null)
        {
            GWM_LOG_STAGE("Bandwidth optimization")
            mBandwidthSizeCriterion = bandwidthSizeCriterionVar(mBandwidthSelectionApproach[i]);
            mBandwidthSelectionCurrentIndex = i;
            mXi = mX.col(i/nRsp);
            mYi = mY.col((i+nRsp)%nRsp);
            mBandwidthSelectionCurrentIndex = i;
            GWM_LOG_INFO(string(GWM_LOG_TAG_GWCORR_INITIAL_BW) + to_string(i));
            BandwidthWeight* bw0 = bandwidth(i);
            BandwidthSelector selector;
            selector.setBandwidth(bw0);
            selector.setLower(bw0->adaptive() ? 20 : 0.0);
            selector.setUpper(bw0->adaptive() ? nDp : mSpatialWeights[i].distance()->maxDistance());
            BandwidthWeight* bw = selector.optimize(this);
            if(bw)
            {
                mSpatialWeights[i].setWeight(bw);
            }
            GWM_LOG_INFO(string(GWM_LOG_TAG_GWCORR_INITIAL_BW)  + to_string(i)  + "," + to_string(bw->bandwidth()));
        }
        GWM_LOG_STOP_RETURN(mStatus, void());
    }

    GWM_LOG_STAGE("Calculating");
    (this->*mSummaryFunction)();
}

// void GWCorrelation::GWCorrelationSerial()
// {
//     mat rankX = mX;
//     rankX.each_col([&](vec &x) { x = rank(x); });
//     uword nVar = mX.n_cols, nRp = mCoords.n_rows;
//     uword corrSize = mIsCorrWithFirstOnly ? 1 : nVar - 1;
//     if (nVar >= 2)
//     {
//         for (uword i = 0; i < nRp; i++)
//         {
//             GWM_LOG_STOP_BREAK(mStatus);
//             vec w = mSpatialWeight.weightVector(i);
//             double sumw = sum(w);
//             vec Wi = w / sumw;
//             mLocalMean.row(i) = trans(Wi) * mX;
//             mat centerized = mX.each_row() - mLocalMean.row(i);
//             mLVar.row(i) = Wi.t() * (centerized % centerized);
//             uword tag = 0;
//             for (uword j = 0; j < corrSize; j++)
//             {
//                 for (uword k = j + 1; k < nVar; k++)
//                 {
//                     double covjk = covwt(mX.col(j), mX.col(k), Wi);
//                     double sumW2 = sum(Wi % Wi);
//                     double covjj = mLVar(i, j) / (1.0 - sumW2);
//                     double covkk = mLVar(i, k) / (1.0 - sumW2);
//                     mCovmat(i, tag) = covjk;
//                     mCorrmat(i, tag) = covjk / sqrt(covjj * covkk);
//                     mSCorrmat(i, tag) = corwt(rankX.col(j), rankX.col(k), Wi);
//                     tag++;
//                 }
//             }
//             GWM_LOG_PROGRESS(i + 1, nRp);
//         }
//     }
//     else{

//         throw std::runtime_error("The number of cols must be 2 or more.");
//     }
// }

// #ifdef ENABLE_OPENMP
// void GWCorrelation::GWCorrelationOmp()
// {
//     mat rankX = mX;
//     rankX.each_col([&](vec &x) { x = rank(x); });
//     uword nVar = mX.n_cols;
//     uword nRp = mCoords.n_rows;
//     uword corrSize = mIsCorrWithFirstOnly ? 1 : nVar - 1;
//     if (nVar >= 2)
//     {
// #pragma omp parallel for num_threads(mOmpThreadNum)
//         for (uword i = 0; i < nRp; i++)
//         {
//             GWM_LOG_STOP_CONTINUE(mStatus);
//             vec w = mSpatialWeight.weightVector(i);
//             double sumw = sum(w);
//             vec Wi = w / sumw;
//             mLocalMean.row(i) = trans(Wi) * mX;
//             mat centerized = mX.each_row() - mLocalMean.row(i);
//             mLVar.row(i) = Wi.t() * (centerized % centerized);
//             uword tag = 0;
//             for (uword j = 0; j < corrSize; j++)
//             {
//                 for (uword k = j + 1; k < nVar; k++)
//                 {
//                     double covjk = covwt(mX.col(j), mX.col(k), Wi);
//                     double sumW2 = sum(Wi % Wi);
//                     double covjj = mLVar(i, j) / (1.0 - sumW2);
//                     double covkk = mLVar(i, k) / (1.0 - sumW2);
//                     mCovmat(i, tag) = covjk;
//                     mCorrmat(i, tag) = covjk / sqrt(covjj * covkk);
//                     mSCorrmat(i, tag) = corwt(rankX.col(j), rankX.col(k), Wi);
//                     tag++;
//                 }
//             }
//             GWM_LOG_PROGRESS(i + 1, nRp);
//         }
//     }
//     else{
//         throw std::runtime_error("The number of cols must be 2 or more.");
//     }
// }
// #endif


// double GWCorrelation::bandwidthSizeCriterionAICSerial(BandwidthWeight *bandwidthWeight)
// {
    // int var = mBandwidthSelectionCurrentIndex;
    // uword nDp = mDataPoints.n_rows;
    // mat betas(1, nDp, fill::zeros);
    // vec shat(2, fill::zeros);
    // for (uword i = 0; i < nDp & !checkCanceled(); i++)
    // {
    //     vec d = mSpatialWeights[var].distance()->distance(i);
    //     vec w = bandwidthWeight->weight(d);
    //     mat xtw = trans(mXi % w);
    //     mat xtwx = xtw * mXi;
    //     mat xtwy = xtw * mYi;
    //     try
    //     {
    //         mat xtwx_inv = inv_sympd(xtwx);
    //         betas.col(i) = xtwx_inv * xtwy;
    //         mat ci = xtwx_inv * xtw;
    //         mat si = mXi(i) * ci;
    //         shat(0) += si(0, i);
    //         shat(1) += det(si * si.t());
    //     }
    //     catch (std::exception e)
    //     {
    //         return DBL_MAX;
    //     }
    // }
    // double value = GwmGWcorrelationTaskThread::AICc(mXi, mYi, betas.t(), shat);
    // if(!checkCanceled()) return value;
    // else return DBL_MAX;
        
    // double value = GWRBase::AICc(mX, mY, betas.t(), shat);
    // if (mStatus == Status::Success && isfinite(value))
    // {
    //     GWM_LOG_PROGRESS_PERCENT(exp(- abs(mBandwidthLastCriterion - value)));
    //     mBandwidthLastCriterion = value;
    //     return value;
    // }
    // else return DBL_MAX;

//     return DBL_MAX;
// }


// double GWCorrelation::bandwidthSizeCriterionCVSerial(BandwidthWeight *bandwidthWeight)
// {
    // int var = mBandwidthSelectionCurrentIndex;
    // uword nDp = mCoords.n_rows;
    // vec shat(2, fill::zeros);
    // double cv = 0.0;
    // for (uword i = 0; i < nDp; i++)
    // {
    //     vec d = mSpatialWeight.distance()->distance(i);
    //     vec w = bandwidthWeight->weight(d);
    //     w(i) = 0.0;
    //     mat xtw = trans(mXi % w);
    //     mat xtwx = xtw * mXi;
    //     mat xtwy = xtw * mYi;
    //     try
    //     {
    //         mat xtwx_inv = inv_sympd(xtwx);
    //         vec beta = xtwx_inv * xtwy;
    //         double res = mYi(i) - det(mXi(i) * beta);
    //         cv += res * res;
    //     }
    //     catch (...)
    //     {
    //         return DBL_MAX;
    //     }
    // }
    // if (mStatus == Status::Success && isfinite(cv))
    // {
    //     GWM_LOG_PROGRESS_PERCENT(exp(- abs(mBandwidthLastCriterion - cv)));
    //     mBandwidthLastCriterion = cv;
    //     return cv;
    // }
    // else return DBL_MAX;

//     return DBL_MAX;
// }
